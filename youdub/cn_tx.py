#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文文本规范化工具（Chinese Text Normalization）

将中文文本中的数字、日期、货币、电话号码、分数、百分数等非标准写法（NSW,
Non-Standard Word）转换为标准中文读音格式。主要用于 TTS（文本转语音）系统
的前端文本处理，确保输入文本能被正确朗读。

核心功能：
  - 数字转中文读音（支持简繁、大写、小写多种形式）
  - 日期规范化（年月日转换为中文读音）
  - 货币金额规范化
  - 电话号码规范化（手机号、固话）
  - 分数、百分数规范化
  - 量词规范化
  - 去除儿化音
  - 全角半角转换
  - 繁简体转换
  - 非法字符检测

作者：
  2019.5 Zhiyang Zhou (https://github.com/Joee1995/chn_text_norm.git)
  2019.9 - 2022 Jiayu DU

使用方式：
  python cn_tx.py [options] input_file output_file

依赖：
  - python 3.X
  - opencc（可选，繁简体转换时需要）
  - loguru（日志记录）

注意：python 2.X 将运行失败或产生错误结果
"""

# ============================================================================ #
# 标准库和第三方库导入
# ============================================================================ #
import sys        # 系统相关，用于命令行参数和标准输出
import os         # 操作系统接口，用于文件路径
import argparse   # 命令行参数解析
import string     # 字符串常量，如标点符号
from loguru import logger  # loguru 日志库
import re         # 正则表达式，用于模式匹配和替换
import csv        # CSV 文件读写，支持 TSV 格式输入

# ============================================================================ #
# 中文数字系统基础常量定义
# ============================================================================ #
# 小写数字（简体）
CHINESE_DIGIS = u'零一二三四五六七八九'
# 大写数字（简体，用于财务/正式场合）
BIG_CHINESE_DIGIS_SIMPLIFIED = u'零壹贰叁肆伍陆柒捌玖'
# 大写数字（繁体）
BIG_CHINESE_DIGIS_TRADITIONAL = u'零壹貳參肆伍陸柒捌玖'

# 小位数单位（简体，十、百、千、万）
SMALLER_BIG_CHINESE_UNITS_SIMPLIFIED = u'十百千万'
# 小位数单位（繁体，拾、佰、仟、萬）
SMALLER_BIG_CHINESE_UNITS_TRADITIONAL = u'拾佰仟萬'

# 大位数单位（简体，亿、兆、京、垓等）
LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED = u'亿兆京垓秭穰沟涧正载'
# 大位数单位（繁体，億、兆、京、垓等）
LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL = u'億兆京垓秭穰溝澗正載'
# 小位数单位别名（简体，与上面相同，用于不同场景）
SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED = u'十百千万'
# 小位数单位别名（繁体）
SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL = u'拾佰仟萬'

# 数字替代写法
ZERO_ALT = u'〇'         # 零的替代写法（用于年份，如"二〇二〇年"）
ONE_ALT = u'幺'          # 一的替代写法（用于电话号码，如"幺幺零"）
TWO_ALTS = [u'两', u'兩']  # 二的替代写法（简体/繁体"两"，用于量词前）

# 符号常量
POSITIVE = [u'正', u'正']     # 正号（简体/繁体）
NEGATIVE = [u'负', u'負']     # 负号（简体/繁体）
POINT = [u'点', u'點']        # 小数点（简体/繁体）

# 填充词（语气词、停顿词等，在语音合成中通常需要去除）
FILLER_CHARS = ['呃', '啊']

# 儿化音白名单正则表达式
# 这些词中的"儿"是合法组成部分，不能被去除
# 例如：女儿、儿子、儿童、婴儿等
ER_WHITELIST = '(儿女|儿子|儿孙|女儿|儿媳|妻儿|' \
    '胎儿|婴儿|新生儿|婴幼儿|幼儿|少儿|小儿|儿歌|儿童|儿科|托儿所|孤儿|' \
    '儿戏|儿化|台儿庄|鹿儿岛|正儿八经|吊儿郎当|生儿育女|托儿带女|养儿防老|痴儿呆女|' \
    '佳儿佳妇|儿怜兽扰|儿无常父|儿不嫌母丑|儿行千里母担忧|儿大不由爷|苏乞儿)'
ER_WHITELIST_PATTERN = re.compile(ER_WHITELIST)

# 中文数字系统类型枚举
# low:  '兆' = '亿' * '十' = 1e9
# mid:  '兆' = '亿' * '万' = 1e12
# high: '兆' = '亿' * '亿' = 1e16
NUMBERING_TYPES = ['low', 'mid', 'high']

# 货币名称（支持的主要货币类型）
CURRENCY_NAMES = '(人民币|美元|日元|英镑|欧元|马克|法郎|加拿大元|澳元|港币|先令|芬兰马克|爱尔兰镑|' \
                 '里拉|荷兰盾|埃斯库多|比塞塔|印尼盾|林吉特|新西兰元|比索|卢布|新加坡元|韩元|泰铢)'
# 货币单位（元、块、角、毛、分等，含数位修饰）
CURRENCY_UNITS = '((亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|)元|(亿|千万|百万|万|千|百|)块|角|毛|分)'
# 常用量词（个、只、条、张、把、次等）
COM_QUANTIFIERS = '(匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|' \
                  '砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|' \
                  '针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|' \
                  '毫|厘|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|' \
                  '盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|旬|' \
                  '纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块)'


# ============================================================================ #
# 中文标点符号常量
# 基于 Zhon 项目 (https://github.com/tsroten/zhon.git)
# ============================================================================ #
CN_PUNCS_STOP = '！？｡。'                    # 中文句末标点（停顿/结束）
CN_PUNCS_NONSTOP = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏·〈〉-'
CN_PUNCS = CN_PUNCS_STOP + CN_PUNCS_NONSTOP  # 全量中文标点

# 中英文标点合集
PUNCS = CN_PUNCS + string.punctuation
# 标点符号转换表：将标点替换为空格
PUNCS_TRANSFORM = str.maketrans(
    PUNCS, ' ' * len(PUNCS), '')  # 将每个标点映射到空格


# ============================================================================ #
# 全角字符到半角字符的转换映射表
# 参考：https://zh.wikipedia.org/wiki/全行和半行
# ============================================================================ #
QJ2BJ = {
    '　': ' ',   # 全角空格
    '！': '!',   # 全角感叹号
    '＂': '"',   # 全角双引号
    '＃': '#',   # 全角井号
    '＄': '$',   # 全角美元符号
    '％': '%',   # 全角百分号
    '＆': '&',   # 全角与符号
    '＇': "'",   # 全角单引号
    '（': '(',   # 全角左括号
    '）': ')',   # 全角右括号
    '＊': '*',   # 全角星号
    '＋': '+',   # 全角加号
    '，': ',',   # 全角逗号
    '－': '-',   # 全角减号/连接线
    '．': '.',   # 全角句点
    '／': '/',   # 全角斜线
    '０': '0',   # 全角数字 0
    '１': '1',   # 全角数字 1
    '２': '2',   # 全角数字 2
    '３': '3',   # 全角数字 3
    '４': '4',   # 全角数字 4
    '５': '5',   # 全角数字 5
    '６': '6',   # 全角数字 6
    '７': '7',   # 全角数字 7
    '８': '8',   # 全角数字 8
    '９': '9',   # 全角数字 9
    '：': ':',   # 全角冒号
    '；': ';',   # 全角分号
    '＜': '<',   # 全角小于号
    '＝': '=',   # 全角等号
    '＞': '>',   # 全角大于号
    '？': '?',   # 全角问号
    '＠': '@',   # 全角 at 符号
    'Ａ': 'A',   # 全角大写 A
    'Ｂ': 'B',   # 全角大写 B
    'Ｃ': 'C',   # 全角大写 C
    'Ｄ': 'D',   # 全角大写 D
    'Ｅ': 'E',   # 全角大写 E
    'Ｆ': 'F',   # 全角大写 F
    'Ｇ': 'G',   # 全角大写 G
    'Ｈ': 'H',   # 全角大写 H
    'Ｉ': 'I',   # 全角大写 I
    'Ｊ': 'J',   # 全角大写 J
    'Ｋ': 'K',   # 全角大写 K
    'Ｌ': 'L',   # 全角大写 L
    'Ｍ': 'M',   # 全角大写 M
    'Ｎ': 'N',   # 全角大写 N
    'Ｏ': 'O',   # 全角大写 O
    'Ｐ': 'P',   # 全角大写 P
    'Ｑ': 'Q',   # 全角大写 Q
    'Ｒ': 'R',   # 全角大写 R
    'Ｓ': 'S',   # 全角大写 S
    'Ｔ': 'T',   # 全角大写 T
    'Ｕ': 'U',   # 全角大写 U
    'Ｖ': 'V',   # 全角大写 V
    'Ｗ': 'W',   # 全角大写 W
    'Ｘ': 'X',   # 全角大写 X
    'Ｙ': 'Y',   # 全角大写 Y
    'Ｚ': 'Z',   # 全角大写 Z
    '［': '[',   # 全角左中括号
    '＼': '\\',  # 全角反斜线
    '］': ']',   # 全角右中括号
    '＾': '^',   # 全角脱字符
    '＿': '_',   # 全角下划线
    '｀': '`',   # 全角反引号
    'ａ': 'a',   # 全角小写 a
    'ｂ': 'b',   # 全角小写 b
    'ｃ': 'c',   # 全角小写 c
    'ｄ': 'd',   # 全角小写 d
    'ｅ': 'e',   # 全角小写 e
    'ｆ': 'f',   # 全角小写 f
    'ｇ': 'g',   # 全角小写 g
    'ｈ': 'h',   # 全角小写 h
    'ｉ': 'i',   # 全角小写 i
    'ｊ': 'j',   # 全角小写 j
    'ｋ': 'k',   # 全角小写 k
    'ｌ': 'l',   # 全角小写 l
    'ｍ': 'm',   # 全角小写 m
    'ｎ': 'n',   # 全角小写 n
    'ｏ': 'o',   # 全角小写 o
    'ｐ': 'p',   # 全角小写 p
    'ｑ': 'q',   # 全角小写 q
    'ｒ': 'r',   # 全角小写 r
    'ｓ': 's',   # 全角小写 s
    'ｔ': 't',   # 全角小写 t
    'ｕ': 'u',   # 全角小写 u
    'ｖ': 'v',   # 全角小写 v
    'ｗ': 'w',   # 全角小写 w
    'ｘ': 'x',   # 全角小写 x
    'ｙ': 'y',   # 全角小写 y
    'ｚ': 'z',   # 全角小写 z
    '｛': '{',   # 全角左大括号
    '｜': '|',   # 全角竖线
    '｝': '}',   # 全角右大括号
    '～': '~',   # 全角波浪号
}
# 创建全角到半角的转换表（用于 str.translate）
QJ2BJ_TRANSFORM = str.maketrans(
    ''.join(QJ2BJ.keys()), ''.join(QJ2BJ.values()), '')


# ============================================================================ #
# 中文字符集常量
# 来源：2013 年中国国家标准《通用规范汉字表》
# 原始资源：https://github.com/mozillazg/pinyin-data/blob/master/kMandarin_8105.txt
# 共 8105 个规范汉字
# ============================================================================ #
CN_CHARS_COMMON = (
    '一丁七万丈三上下不与丏丐丑专且丕世丘丙业丛东丝丞丢两严丧个丫中丰串临丸丹为主丽举'
    '乂乃久么义之乌乍乎乏乐乒乓乔乖乘乙乜九乞也习乡书乩买乱乳乸乾了予争事二亍于亏云互'
    '亓五井亘亚些亟亡亢交亥亦产亨亩享京亭亮亲亳亵亶亸亹人亿什仁仂仃仄仅仆仇仉今介仍从'
    '仑仓仔仕他仗付仙仝仞仟仡代令以仨仪仫们仰仲仳仵件价任份仿企伈伉伊伋伍伎伏伐休众优'
    '伙会伛伞伟传伢伣伤伥伦伧伪伫伭伯估伲伴伶伸伺似伽伾佁佃但位低住佐佑体何佖佗佘余佚'
    '佛作佝佞佟你佣佤佥佩佬佯佰佳佴佶佸佺佻佼佽佾使侁侂侃侄侈侉例侍侏侑侔侗侘供依侠侣'
    '侥侦侧侨侩侪侬侮侯侴侵侹便促俄俅俊俍俎俏俐俑俗俘俙俚俜保俞俟信俣俦俨俩俪俫俭修俯'
    '俱俳俵俶俸俺俾倌倍倏倒倓倔倕倘候倚倜倞借倡倥倦倧倨倩倪倬倭倮倴债倻值倾偁偃假偈偌'
    '偎偏偓偕做停偡健偬偭偰偲偶偷偻偾偿傀傃傅傈傉傍傒傕傣傥傧储傩催傲傺傻僇僎像僔僖僚'
    '僦僧僬僭僮僰僳僵僻儆儇儋儒儡儦儳儴儿兀允元兄充兆先光克免兑兔兕兖党兜兢入全八公六'
    '兮兰共关兴兵其具典兹养兼兽冀冁内冈冉册再冏冒冔冕冗写军农冠冢冤冥冬冮冯冰冱冲决况'
    '冶冷冻冼冽净凄准凇凉凋凌减凑凓凘凛凝几凡凤凫凭凯凰凳凶凸凹出击凼函凿刀刁刃分切刈'
    '刊刍刎刑划刖列刘则刚创初删判刨利别刬刭刮到刳制刷券刹刺刻刽刿剀剁剂剃剅削剋剌前剐'
    '剑剔剕剖剜剞剟剡剥剧剩剪副割剽剿劁劂劄劈劐劓力劝办功加务劢劣动助努劫劬劭励劲劳劼'
    '劾势勃勇勉勋勍勐勒勔勖勘勚募勠勤勰勺勾勿匀包匆匈匍匏匐匕化北匙匜匝匠匡匣匦匪匮匹'
    '区医匼匾匿十千卅升午卉半华协卑卒卓单卖南博卜卞卟占卡卢卣卤卦卧卫卬卮卯印危即却卵'
    '卷卸卺卿厂厄厅历厉压厌厍厕厖厘厚厝原厢厣厥厦厨厩厮去厾县叁参叆叇又叉及友双反发叔'
    '叕取受变叙叚叛叟叠口古句另叨叩只叫召叭叮可台叱史右叵叶号司叹叻叼叽吁吃各吆合吉吊'
    '同名后吏吐向吒吓吕吖吗君吝吞吟吠吡吣否吧吨吩含听吭吮启吱吲吴吵吸吹吻吼吽吾呀呃呆'
    '呇呈告呋呐呒呓呔呕呖呗员呙呛呜呢呣呤呦周呱呲味呵呶呷呸呻呼命咀咂咄咆咇咉咋和咍咎'
    '咏咐咒咔咕咖咙咚咛咝咡咣咤咥咦咧咨咩咪咫咬咯咱咳咴咸咺咻咽咿哀品哂哃哄哆哇哈哉哌'
    '响哎哏哐哑哒哓哔哕哗哙哚哝哞哟哢哥哦哧哨哩哪哭哮哱哲哳哺哼哽哿唁唆唇唉唏唐唑唔唛'
    '唝唠唢唣唤唧唪唬售唯唰唱唳唵唷唼唾唿啁啃啄商啉啊啐啕啖啜啡啤啥啦啧啪啫啬啭啮啰啴'
    '啵啶啷啸啻啼啾喀喁喂喃善喆喇喈喉喊喋喏喑喔喘喙喜喝喟喤喧喱喳喵喷喹喻喽喾嗄嗅嗉嗌'
    '嗍嗐嗑嗒嗓嗔嗖嗜嗝嗞嗟嗡嗣嗤嗥嗦嗨嗪嗫嗬嗯嗲嗳嗵嗷嗽嗾嘀嘁嘈嘉嘌嘎嘏嘘嘚嘛嘞嘟嘡'
    '嘣嘤嘧嘬嘭嘱嘲嘴嘶嘹嘻嘿噀噂噇噌噍噎噔噗噘噙噜噢噤器噩噪噫噬噱噶噻噼嚄嚅嚆嚎嚏嚓'
    '嚚嚣嚭嚯嚷嚼囊囔囚四回囟因囡团囤囫园困囱围囵囷囹固国图囿圃圄圆圈圉圊圌圐圙圜土圢'
    '圣在圩圪圫圬圭圮圯地圲圳圹场圻圾址坂均坉坊坋坌坍坎坏坐坑坒块坚坛坜坝坞坟坠坡坤坥'
    '坦坨坩坪坫坬坭坯坰坳坷坻坼坽垂垃垄垆垈型垌垍垎垏垒垓垕垙垚垛垞垟垠垡垢垣垤垦垧垩'
    '垫垭垮垯垱垲垴垵垸垺垾垿埂埃埆埇埋埌城埏埒埔埕埗埘埙埚埝域埠埤埪埫埭埯埴埵埸培基'
    '埼埽堂堃堆堇堉堋堌堍堎堐堑堕堙堞堠堡堤堧堨堪堰堲堵堼堽堾塄塅塆塌塍塑塔塘塝塞塥填'
    '塬塱塾墀墁境墅墈墉墐墒墓墕墘墙墚增墟墡墣墦墨墩墼壁壅壑壕壤士壬壮声壳壶壸壹处备复'
    '夏夐夔夕外夙多夜够夤夥大天太夫夬夭央夯失头夷夸夹夺夼奁奂奄奇奈奉奋奎奏契奓奔奕奖'
    '套奘奚奠奡奢奥奭女奴奶奸她好妁如妃妄妆妇妈妊妍妒妓妖妗妘妙妞妣妤妥妧妨妩妪妫妭妮'
    '妯妲妹妻妾姆姈姊始姐姑姒姓委姗姘姚姜姝姞姣姤姥姨姬姮姱姶姹姻姽姿娀威娃娄娅娆娇娈'
    '娉娌娑娓娘娜娟娠娣娥娩娱娲娴娵娶娼婀婆婉婊婌婍婕婘婚婞婠婢婤婧婪婫婳婴婵婶婷婺婻'
    '婼婿媂媄媆媒媓媖媚媛媞媪媭媱媲媳媵媸媾嫁嫂嫄嫉嫌嫒嫔嫕嫖嫘嫚嫜嫠嫡嫣嫦嫩嫪嫫嫭嫱'
    '嫽嬉嬖嬗嬛嬥嬬嬴嬷嬿孀孅子孑孓孔孕孖字存孙孚孛孜孝孟孢季孤孥学孩孪孬孰孱孳孵孺孽'
    '宁它宄宅宇守安宋完宏宓宕宗官宙定宛宜宝实宠审客宣室宥宦宧宪宫宬宰害宴宵家宸容宽宾'
    '宿寁寂寄寅密寇富寐寒寓寝寞察寡寤寥寨寮寰寸对寺寻导寿封射将尉尊小少尔尕尖尘尚尜尝'
    '尢尤尥尧尨尪尬就尴尸尹尺尻尼尽尾尿局屁层屃居屈屉届屋屎屏屐屑展屙属屠屡屣履屦屯山'
    '屹屺屼屾屿岁岂岈岊岌岍岐岑岔岖岗岘岙岚岛岜岞岠岢岣岨岩岫岬岭岱岳岵岷岸岽岿峁峂峃'
    '峄峋峒峗峘峙峛峡峣峤峥峦峧峨峪峭峰峱峻峿崀崁崂崃崄崆崇崌崎崒崔崖崚崛崞崟崡崤崦崧'
    '崩崭崮崴崶崽崾崿嵁嵅嵇嵊嵋嵌嵎嵖嵘嵚嵛嵝嵩嵫嵬嵯嵲嵴嶂嶅嶍嶒嶓嶙嶝嶟嶦嶲嶷巅巇巉'
    '巍川州巡巢工左巧巨巩巫差巯己已巳巴巷巽巾币市布帅帆师希帏帐帑帔帕帖帘帙帚帛帜帝帡'
    '带帧帨席帮帱帷常帻帼帽幂幄幅幌幔幕幖幛幞幡幢幪干平年并幸幺幻幼幽广庄庆庇床庋序庐'
    '庑库应底庖店庙庚府庞废庠庤庥度座庭庱庳庵庶康庸庹庼庾廆廉廊廋廑廒廓廖廙廛廨廪延廷'
    '建廿开弁异弃弄弆弇弈弊弋式弑弓引弗弘弛弟张弢弥弦弧弨弩弭弯弱弶弸弹强弼彀归当录彖'
    '彗彘彝彟形彤彦彧彩彪彬彭彰影彳彷役彻彼往征徂径待徇很徉徊律徐徒徕得徘徙徛徜御徨循'
    '徭微徵德徼徽心必忆忉忌忍忏忐忑忒忖志忘忙忝忞忠忡忤忧忪快忭忮忱忳念忸忺忻忽忾忿怀'
    '态怂怃怄怅怆怊怍怎怏怒怔怕怖怙怛怜思怠怡急怦性怨怩怪怫怯怵总怼怿恁恂恃恋恍恐恒恓'
    '恔恕恙恚恝恢恣恤恧恨恩恪恫恬恭息恰恳恶恸恹恺恻恼恽恿悃悄悆悈悉悌悍悒悔悖悚悛悝悟'
    '悠悢患悦您悫悬悭悯悰悱悲悴悸悻悼情惆惇惊惋惎惑惔惕惘惙惚惛惜惝惟惠惦惧惨惩惫惬惭'
    '惮惯惰想惴惶惹惺愀愁愃愆愈愉愍愎意愐愔愕愚感愠愣愤愦愧愫愭愿慆慈慊慌慎慑慕慝慢慥'
    '慧慨慬慭慰慵慷憋憎憔憕憙憧憨憩憬憭憷憺憾懂懈懊懋懑懒懔懦懵懿戆戈戊戋戌戍戎戏成我'
    '戒戕或戗战戚戛戟戡戢戣戤戥截戬戭戮戳戴户戽戾房所扁扂扃扅扆扇扈扉扊手才扎扑扒打扔'
    '托扛扞扣扦执扩扪扫扬扭扮扯扰扳扶批扺扼扽找承技抃抄抉把抑抒抓抔投抖抗折抚抛抟抠抡'
    '抢护报抨披抬抱抵抹抻押抽抿拂拃拄担拆拇拈拉拊拌拍拎拐拒拓拔拖拗拘拙招拜拟拢拣拤拥'
    '拦拧拨择括拭拮拯拱拳拴拶拷拼拽拾拿持挂指挈按挎挑挓挖挚挛挝挞挟挠挡挣挤挥挦挨挪挫'
    '振挲挹挺挽捂捃捅捆捉捋捌捍捎捏捐捕捞损捡换捣捧捩捭据捯捶捷捺捻捽掀掂掇授掉掊掌掎'
    '掏掐排掖掘掞掠探掣接控推掩措掬掭掮掰掳掴掷掸掺掼掾揄揆揉揍描提插揕揖揠握揣揩揪揭'
    '揳援揶揸揽揿搀搁搂搅搋搌搏搐搒搓搔搛搜搞搠搡搦搪搬搭搴携搽摁摄摅摆摇摈摊摏摒摔摘'
    '摛摞摧摩摭摴摸摹摽撂撄撅撇撑撒撕撖撙撞撤撩撬播撮撰撵撷撸撺撼擀擂擅操擎擐擒擘擞擢'
    '擤擦擿攀攉攒攘攥攫攮支收攸改攻攽放政故效敉敌敏救敔敕敖教敛敝敞敢散敦敩敫敬数敲整'
    '敷文斋斌斐斑斓斗料斛斜斝斟斠斡斤斥斧斩斫断斯新斶方於施旁旃旄旅旆旋旌旎族旐旒旖旗'
    '旞无既日旦旧旨早旬旭旮旯旰旱旴旵时旷旸旺旻旿昀昂昃昄昆昇昈昉昊昌明昏昒易昔昕昙昝'
    '星映昡昣昤春昧昨昪昫昭是昱昳昴昵昶昺昼昽显晁晃晅晊晋晌晏晐晒晓晔晕晖晗晙晚晞晟晡'
    '晢晤晦晨晪晫普景晰晱晴晶晷智晾暂暄暅暇暌暑暕暖暗暝暧暨暮暲暴暵暶暹暾暿曈曌曙曛曜'
    '曝曦曩曰曲曳更曷曹曼曾替最月有朋服朏朐朓朔朕朗望朝期朦木未末本札术朱朳朴朵朸机朽'
    '杀杂权杄杆杈杉杌李杏材村杓杕杖杙杜杞束杠条来杧杨杩杪杭杯杰杲杳杵杷杻杼松板极构枅'
    '枇枉枋枍析枕林枘枚果枝枞枢枣枥枧枨枪枫枭枯枰枲枳枵架枷枸枹柁柃柄柈柊柏某柑柒染柔'
    '柖柘柙柚柜柝柞柠柢查柩柬柯柰柱柳柴柷柽柿栀栅标栈栉栊栋栌栎栏栐树栒栓栖栗栝栟校栩'
    '株栲栳栴样核根栻格栽栾桀桁桂桃桄桅框案桉桊桌桎桐桑桓桔桕桠桡桢档桤桥桦桧桨桩桫桯'
    '桲桴桶桷桹梁梃梅梆梌梏梓梗梠梢梣梦梧梨梭梯械梳梴梵梼梽梾梿检棁棂棉棋棍棐棒棓棕棘'
    '棚棠棣棤棨棪棫棬森棰棱棵棹棺棻棼棽椀椁椅椆椋植椎椐椑椒椓椟椠椤椪椭椰椴椸椹椽椿楂'
    '楒楔楗楙楚楝楞楠楣楦楩楪楫楮楯楷楸楹楼概榃榄榅榆榇榈榉榍榑榔榕榖榛榜榧榨榫榭榰榱'
    '榴榷榻槁槃槊槌槎槐槔槚槛槜槟槠槭槱槲槽槿樊樗樘樟模樨横樯樱樵樽樾橄橇橐橑橘橙橛橞'
    '橡橥橦橱橹橼檀檄檎檐檑檗檞檠檩檫檬櫆欂欠次欢欣欤欧欲欸欹欺欻款歃歅歆歇歉歌歙止正'
    '此步武歧歪歹死歼殁殂殃殄殆殇殉殊残殍殒殓殖殚殛殡殣殪殳殴段殷殿毁毂毅毋毌母每毐毒'
    '毓比毕毖毗毙毛毡毪毫毯毳毵毹毽氅氆氇氍氏氐民氓气氕氖氘氙氚氛氟氡氢氤氦氧氨氩氪氮'
    '氯氰氲水永氾氿汀汁求汆汇汈汉汊汋汐汔汕汗汛汜汝汞江池污汤汧汨汩汪汫汭汰汲汴汶汹汽'
    '汾沁沂沃沄沅沆沇沈沉沌沏沐沓沔沘沙沚沛沟没沣沤沥沦沧沨沩沪沫沭沮沱河沸油沺治沼沽'
    '沾沿泂泃泄泅泇泉泊泌泐泓泔法泖泗泙泚泛泜泞泠泡波泣泥注泪泫泮泯泰泱泳泵泷泸泺泻泼'
    '泽泾洁洄洇洈洋洌洎洑洒洓洗洘洙洚洛洞洢洣津洧洨洪洫洭洮洱洲洳洴洵洸洹洺活洼洽派洿'
    '流浃浅浆浇浈浉浊测浍济浏浐浑浒浓浔浕浙浚浛浜浞浟浠浡浣浥浦浩浪浬浭浮浯浰浲浴海浸'
    '浼涂涄涅消涉涌涍涎涐涑涓涔涕涘涛涝涞涟涠涡涢涣涤润涧涨涩涪涫涮涯液涴涵涸涿淀淄淅'
    '淆淇淋淌淏淑淖淘淙淜淝淞淟淠淡淤淦淫淬淮淯深淳淴混淹添淼清渊渌渍渎渐渑渔渗渚渝渟'
    '渠渡渣渤渥温渫渭港渰渲渴游渺渼湃湄湉湍湎湑湓湔湖湘湛湜湝湟湣湫湮湲湴湾湿溁溃溅溆'
    '溇溉溍溏源溘溚溜溞溟溠溢溥溦溧溪溯溱溲溴溵溶溷溹溺溻溽滁滂滃滆滇滉滋滍滏滑滓滔滕'
    '滗滘滚滞滟滠满滢滤滥滦滧滨滩滪滫滴滹漂漆漈漉漋漏漓演漕漖漠漤漦漩漪漫漭漯漱漳漴漶'
    '漷漹漻漼漾潆潇潋潍潏潖潘潜潞潟潢潦潩潭潮潲潴潵潸潺潼潽潾澂澄澈澉澌澍澎澛澜澡澥澧'
    '澪澭澳澴澶澹澼澽激濂濉濋濑濒濞濠濡濩濮濯瀌瀍瀑瀔瀚瀛瀣瀱瀵瀹瀼灈灌灏灞火灭灯灰灵'
    '灶灸灼灾灿炀炅炆炉炊炌炎炒炔炕炖炘炙炜炝炟炣炫炬炭炮炯炱炳炷炸点炻炼炽烀烁烂烃烈'
    '烊烔烘烙烛烜烝烟烠烤烦烧烨烩烫烬热烯烶烷烹烺烻烽焆焉焊焌焐焓焕焖焗焘焙焚焜焞焦焯'
    '焰焱然煁煃煅煊煋煌煎煓煜煞煟煤煦照煨煮煲煳煴煸煺煽熄熇熊熏熔熘熙熛熜熟熠熥熨熬熵'
    '熹熻燃燊燋燎燏燔燕燚燠燥燧燮燹爆爇爔爚爝爟爨爪爬爰爱爵父爷爸爹爻爽爿牁牂片版牌牍'
    '牒牖牙牚牛牝牟牡牢牤牥牦牧物牮牯牲牵特牺牻牾牿犀犁犄犇犊犋犍犏犒犟犨犬犯犰犴状犷'
    '犸犹狁狂狃狄狈狉狍狎狐狒狗狙狝狞狠狡狨狩独狭狮狯狰狱狲狳狴狷狸狺狻狼猁猃猄猇猊猎'
    '猕猖猗猛猜猝猞猡猢猥猩猪猫猬献猯猰猱猴猷猹猺猾猿獍獐獒獗獠獬獭獯獴獾玃玄率玉王玎'
    '玑玒玓玕玖玘玙玚玛玞玟玠玡玢玤玥玦玩玫玭玮环现玱玲玳玶玷玹玺玻玼玿珀珂珅珇珈珉珊'
    '珋珌珍珏珐珑珒珕珖珙珛珝珞珠珢珣珥珦珧珩珪珫班珰珲珵珷珸珹珺珽琀球琄琅理琇琈琉琊'
    '琎琏琐琔琚琛琟琡琢琤琥琦琨琪琫琬琭琮琯琰琲琳琴琵琶琼瑀瑁瑂瑃瑄瑅瑆瑑瑓瑔瑕瑖瑗瑙'
    '瑚瑛瑜瑝瑞瑟瑢瑧瑨瑬瑭瑰瑱瑳瑶瑷瑾璀璁璃璆璇璈璋璎璐璒璘璜璞璟璠璥璧璨璩璪璬璮璱'
    '璲璺瓀瓒瓖瓘瓜瓞瓠瓢瓣瓤瓦瓮瓯瓴瓶瓷瓻瓿甄甍甏甑甓甗甘甚甜生甡甥甦用甩甪甫甬甭甯'
    '田由甲申电男甸町画甾畀畅畈畋界畎畏畔畖留畚畛畜畤略畦番畬畯畲畴畸畹畿疁疃疆疍疏疐'
    '疑疔疖疗疙疚疝疟疠疡疢疣疤疥疫疬疭疮疯疰疱疲疳疴疵疸疹疼疽疾痂痃痄病症痈痉痊痍痒'
    '痓痔痕痘痛痞痢痣痤痦痧痨痪痫痰痱痴痹痼痿瘀瘁瘃瘅瘆瘊瘌瘐瘕瘗瘘瘙瘛瘟瘠瘢瘤瘥瘦瘩'
    '瘪瘫瘭瘰瘳瘴瘵瘸瘼瘾瘿癀癃癌癍癔癖癗癜癞癣癫癯癸登白百癿皂的皆皇皈皋皎皑皓皕皖皙'
    '皛皞皤皦皭皮皱皲皴皿盂盅盆盈盉益盍盎盏盐监盒盔盖盗盘盛盟盥盦目盯盱盲直盷相盹盼盾'
    '省眄眇眈眉眊看眍眙眚真眠眢眦眨眩眬眭眯眵眶眷眸眺眼着睁睃睄睇睎睐睑睚睛睡睢督睥睦'
    '睨睫睬睹睽睾睿瞀瞄瞅瞋瞌瞍瞎瞑瞒瞟瞠瞢瞥瞧瞩瞪瞫瞬瞭瞰瞳瞵瞻瞽瞿矍矗矛矜矞矢矣知'
    '矧矩矫矬短矮矰石矶矸矻矼矾矿砀码砂砄砆砉砌砍砑砒研砖砗砘砚砜砝砟砠砣砥砧砫砬砭砮'
    '砰破砵砷砸砹砺砻砼砾础硁硅硇硊硌硍硎硐硒硔硕硖硗硙硚硝硪硫硬硭确硼硿碃碇碈碉碌碍'
    '碎碏碑碓碗碘碚碛碜碟碡碣碥碧碨碰碱碲碳碴碶碹碾磁磅磉磊磋磏磐磔磕磙磜磡磨磬磲磴磷'
    '磹磻礁礅礌礓礞礴礵示礼社祀祁祃祆祇祈祉祊祋祎祏祐祓祕祖祗祚祛祜祝神祟祠祢祥祧票祭'
    '祯祲祷祸祺祼祾禀禁禄禅禊禋福禒禔禘禚禛禤禧禳禹禺离禽禾秀私秃秆秉秋种科秒秕秘租秣'
    '秤秦秧秩秫秬秭积称秸移秽秾稀稂稃稆程稌稍税稑稔稗稙稚稞稠稣稳稷稹稻稼稽稿穄穆穑穗'
    '穙穜穟穰穴究穷穸穹空穿窀突窃窄窅窈窊窍窎窑窒窕窖窗窘窜窝窟窠窣窥窦窨窬窭窳窸窿立'
    '竑竖竘站竞竟章竣童竦竫竭端竹竺竽竿笃笄笆笈笊笋笏笑笔笕笙笛笞笠笤笥符笨笪笫第笮笯'
    '笱笳笸笺笼笾筀筅筇等筋筌筏筐筑筒答策筘筚筛筜筝筠筢筤筥筦筮筱筲筵筶筷筹筻筼签简箅'
    '箍箐箓箔箕箖算箜管箢箦箧箨箩箪箫箬箭箱箴箸篁篆篇篌篑篓篙篚篝篡篥篦篪篮篯篱篷篼篾'
    '簃簇簉簋簌簏簕簖簝簟簠簧簪簰簸簿籀籁籍籥米籴类籼籽粉粑粒粕粗粘粜粝粞粟粢粤粥粪粮'
    '粱粲粳粹粼粽精粿糁糅糇糈糊糌糍糒糕糖糗糙糜糟糠糨糯糵系紊素索紧紫累絜絮絷綦綮縠縢'
    '縻繁繄繇纂纛纠纡红纣纤纥约级纨纩纪纫纬纭纮纯纰纱纲纳纴纵纶纷纸纹纺纻纼纽纾线绀绁'
    '绂练组绅细织终绉绊绋绌绍绎经绐绑绒结绔绕绖绗绘给绚绛络绝绞统绠绡绢绣绤绥绦继绨绩'
    '绪绫续绮绯绰绱绲绳维绵绶绷绸绹绺绻综绽绾绿缀缁缂缃缄缅缆缇缈缉缊缌缎缐缑缒缓缔缕'
    '编缗缘缙缚缛缜缝缞缟缠缡缢缣缤缥缦缧缨缩缪缫缬缭缮缯缰缱缲缳缴缵缶缸缺罂罄罅罍罐'
    '网罔罕罗罘罚罟罡罢罨罩罪置罱署罴罶罹罽罾羁羊羌美羑羓羔羕羖羚羝羞羟羡群羧羯羰羱羲'
    '羸羹羼羽羿翀翁翂翃翅翈翊翌翎翔翕翘翙翚翛翟翠翡翥翦翩翮翯翰翱翳翷翻翼翾耀老考耄者'
    '耆耇耋而耍耏耐耑耒耔耕耖耗耘耙耜耠耢耤耥耦耧耨耩耪耰耱耳耵耶耷耸耻耽耿聂聃聆聊聋'
    '职聍聒联聘聚聩聪聱聿肃肄肆肇肉肋肌肓肖肘肚肛肝肟肠股肢肤肥肩肪肫肭肮肯肱育肴肷肸'
    '肺肼肽肾肿胀胁胂胃胄胆胈背胍胎胖胗胙胚胛胜胝胞胠胡胣胤胥胧胨胩胪胫胬胭胯胰胱胲胳'
    '胴胶胸胺胼能脂脆脉脊脍脎脏脐脑脒脓脔脖脘脚脞脟脩脬脯脱脲脶脸脾脿腆腈腊腋腌腐腑腒'
    '腓腔腕腘腙腚腠腥腧腨腩腭腮腯腰腱腴腹腺腻腼腽腾腿膀膂膈膊膏膑膘膙膛膜膝膦膨膳膺膻'
    '臀臂臃臆臊臌臑臜臣臧自臬臭至致臻臼臾舀舁舂舄舅舆舌舍舐舒舔舛舜舞舟舠舢舣舥航舫般'
    '舭舯舰舱舲舳舴舵舶舷舸船舻舾艄艅艇艉艋艎艏艘艚艟艨艮良艰色艳艴艺艽艾艿节芃芄芈芊'
    '芋芍芎芏芑芒芗芘芙芜芝芟芠芡芣芤芥芦芨芩芪芫芬芭芮芯芰花芳芴芷芸芹芼芽芾苁苄苇苈'
    '苉苊苋苌苍苎苏苑苒苓苔苕苗苘苛苜苞苟苠苡苣苤若苦苧苫苯英苴苷苹苻苾茀茁茂范茄茅茆'
    '茈茉茋茌茎茏茑茓茔茕茗茚茛茜茝茧茨茫茬茭茯茱茳茴茵茶茸茹茺茼茽荀荁荃荄荆荇草荏荐'
    '荑荒荓荔荖荙荚荛荜荞荟荠荡荣荤荥荦荧荨荩荪荫荬荭荮药荷荸荻荼荽莅莆莉莎莒莓莘莙莛'
    '莜莝莞莠莨莩莪莫莰莱莲莳莴莶获莸莹莺莼莽莿菀菁菂菅菇菉菊菌菍菏菔菖菘菜菝菟菠菡菥'
    '菩菪菰菱菲菹菼菽萁萃萄萆萋萌萍萎萏萑萘萚萜萝萣萤营萦萧萨萩萱萳萸萹萼落葆葎葑葖著'
    '葙葚葛葜葡董葩葫葬葭葰葱葳葴葵葶葸葺蒂蒄蒇蒈蒉蒋蒌蒎蒐蒗蒙蒜蒟蒡蒨蒯蒱蒲蒴蒸蒹蒺'
    '蒻蒽蒿蓁蓂蓄蓇蓉蓊蓍蓏蓐蓑蓓蓖蓝蓟蓠蓢蓣蓥蓦蓬蓰蓼蓿蔀蔃蔈蔊蔌蔑蔓蔗蔚蔟蔡蔫蔬蔷'
    '蔸蔹蔺蔻蔼蔽蕃蕈蕉蕊蕖蕗蕙蕞蕤蕨蕰蕲蕴蕹蕺蕻蕾薁薄薅薇薏薛薜薢薤薨薪薮薯薰薳薷薸'
    '薹薿藁藉藏藐藓藕藜藟藠藤藦藨藩藻藿蘅蘑蘖蘘蘧蘩蘸蘼虎虏虐虑虒虓虔虚虞虢虤虫虬虮虱'
    '虷虸虹虺虻虼虽虾虿蚀蚁蚂蚄蚆蚊蚋蚌蚍蚓蚕蚜蚝蚣蚤蚧蚨蚩蚪蚬蚯蚰蚱蚲蚴蚶蚺蛀蛃蛄蛆'
    '蛇蛉蛊蛋蛎蛏蛐蛑蛔蛘蛙蛛蛞蛟蛤蛩蛭蛮蛰蛱蛲蛳蛴蛸蛹蛾蜀蜂蜃蜇蜈蜉蜊蜍蜎蜐蜒蜓蜕蜗'
    '蜘蜚蜜蜞蜡蜢蜣蜥蜩蜮蜱蜴蜷蜻蜾蜿蝇蝈蝉蝌蝎蝓蝗蝘蝙蝠蝣蝤蝥蝮蝰蝲蝴蝶蝻蝼蝽蝾螂螃'
    '螅螈螋融螗螟螠螣螨螫螬螭螯螱螳螵螺螽蟀蟆蟊蟋蟏蟑蟒蟛蟠蟥蟪蟫蟮蟹蟾蠃蠊蠋蠓蠕蠖蠡'
    '蠢蠲蠹蠼血衃衄衅行衍衎衒衔街衙衠衡衢衣补表衩衫衬衮衰衲衷衽衾衿袁袂袄袅袆袈袋袍袒'
    '袖袗袜袢袤袪被袭袯袱袷袼裁裂装裆裈裉裎裒裔裕裘裙裛裟裢裣裤裥裨裰裱裳裴裸裹裼裾褂'
    '褊褐褒褓褕褙褚褛褟褡褥褪褫褯褰褴褶襁襄襕襚襜襞襟襦襫襻西要覃覆见观觃规觅视觇览觉'
    '觊觋觌觎觏觐觑角觖觚觜觞觟解觥触觫觭觯觱觳觿言訄訇訚訾詈詟詹誉誊誓謇警譬计订讣认'
    '讥讦讧讨让讪讫训议讯记讱讲讳讴讵讶讷许讹论讻讼讽设访诀证诂诃评诅识诇诈诉诊诋诌词'
    '诎诏诐译诒诓诔试诖诗诘诙诚诛诜话诞诟诠诡询诣诤该详诧诨诩诫诬语诮误诰诱诲诳说诵请'
    '诸诹诺读诼诽课诿谀谁谂调谄谅谆谇谈谊谋谌谍谎谏谐谑谒谓谔谕谖谗谙谚谛谜谝谞谟谠谡'
    '谢谣谤谥谦谧谨谩谪谫谬谭谮谯谰谱谲谳谴谵谶谷谼谿豁豆豇豉豌豕豚象豢豨豪豫豮豳豸豹'
    '豺貂貅貆貉貊貌貔貘贝贞负贡财责贤败账货质贩贪贫贬购贮贯贰贱贲贳贴贵贶贷贸费贺贻贼'
    '贽贾贿赀赁赂赃资赅赆赇赈赉赊赋赌赍赎赏赐赑赒赓赔赕赖赗赘赙赚赛赜赝赞赟赠赡赢赣赤'
    '赦赧赪赫赭走赳赴赵赶起趁趄超越趋趑趔趟趣趯趱足趴趵趸趺趼趾趿跂跃跄跆跋跌跎跏蹐跑'
    '跖跗跚跛距跞跟跣跤跨跪跬路跱跳践跶跷跸跹跺跻跽踅踉踊踌踏踒踔踝踞踟踢踣踦踩踪踬踮'
    '踯踱踵踶踹踺踽蹀蹁蹂蹄蹅蹇蹈蹉蹊蹋蹐蹑蹒蹙蹚蹜蹢蹦蹩蹬蹭蹯蹰蹲蹴蹶蹼蹽蹾蹿躁躅躇'
    '躏躐躔躜躞身躬躯躲躺车轧轨轩轪轫转轭轮软轰轱轲轳轴轵轶轷轸轹轺轻轼载轾轿辀辁辂较'
    '辄辅辆辇辈辉辊辋辌辍辎辏辐辑辒输辔辕辖辗辘辙辚辛辜辞辟辣辨辩辫辰辱边辽达辿迁迂迄'
    '迅过迈迎运近迓返迕还这进远违连迟迢迤迥迦迨迩迪迫迭迮述迳迷迸迹迺追退送适逃逄逅逆'
    '选逊逋逍透逐逑递途逖逗通逛逝逞速造逡逢逦逭逮逯逴逵逶逸逻逼逾遁遂遄遆遇遍遏遐遑遒'
    '道遗遘遛遢遣遥遨遭遮遴遵遹遽避邀邂邃邈邋邑邓邕邗邘邙邛邝邠邡邢那邦邨邪邬邮邯邰邱'
    '邲邳邴邵邶邸邹邺邻邽邾邿郁郃郄郅郇郈郊郎郏郐郑郓郗郚郛郜郝郡郢郤郦郧部郪郫郭郯郴'
    '郸都郾郿鄀鄂鄃鄄鄅鄌鄑鄗鄘鄙鄚鄜鄞鄠鄢鄣鄫鄯鄱鄹酂酃酅酆酉酊酋酌配酎酏酐酒酗酚酝'
    '酞酡酢酣酤酥酦酩酪酬酮酯酰酱酲酴酵酶酷酸酹酺酽酾酿醅醇醉醋醌醍醐醑醒醚醛醢醨醪醭'
    '醮醯醴醵醺醾采釉释里重野量釐金釜鉴銎銮鋆鋈錾鍪鎏鏊鏖鐾鑫钆钇针钉钊钋钌钍钎钏钐钒'
    '钓钔钕钖钗钘钙钚钛钜钝钞钟钠钡钢钣钤钥钦钧钨钩钪钫钬钭钮钯钰钱钲钳钴钵钷钹钺钻钼'
    '钽钾钿铀铁铂铃铄铅铆铈铉铊铋铌铍铎铏铐铑铒铕铖铗铘铙铚铛铜铝铞铟铠铡铢铣铤铥铧铨'
    '铩铪铫铬铭铮铯铰铱铲铳铴铵银铷铸铹铺铻铼铽链铿销锁锂锃锄锅锆锇锈锉锊锋锌锍锎锏锐'
    '锑锒锓锔锕锖锗锘错锚锛锜锝锞锟锡锢锣锤锥锦锧锨锩锪锫锬锭键锯锰锱锲锳锴锵锶锷锸锹'
    '锺锻锼锽锾锿镀镁镂镃镄镅镆镇镈镉镊镋镌镍镎镏镐镑镒镓镔镕镖镗镘镚镛镜镝镞镠镡镢镣'
    '镤镥镦镧镨镩镪镫镬镭镮镯镰镱镲镳镴镵镶长门闩闪闫闭问闯闰闱闲闳间闵闶闷闸闹闺闻闼'
    '闽闾闿阀阁阂阃阄阅阆阇阈阉阊阋阌阍阎阏阐阑阒阔阕阖阗阘阙阚阜队阡阪阮阱防阳阴阵阶'
    '阻阼阽阿陀陂附际陆陇陈陉陋陌降陎限陑陔陕陛陞陟陡院除陧陨险陪陬陲陴陵陶陷隃隅隆隈'
    '隋隍随隐隔隗隘隙障隧隩隰隳隶隹隺隼隽难雀雁雄雅集雇雉雊雌雍雎雏雒雕雠雨雩雪雯雱雳'
    '零雷雹雾需霁霄霅霆震霈霉霍霎霏霓霖霜霞霨霪霭霰露霸霹霾青靓靖静靛非靠靡面靥革靬靰'
    '靳靴靶靸靺靼靽靿鞁鞅鞋鞍鞑鞒鞔鞘鞠鞡鞣鞧鞨鞫鞬鞭鞮鞯鞲鞳鞴韂韦韧韨韩韪韫韬韭音韵'
    '韶页顶顷顸项顺须顼顽顾顿颀颁颂颃预颅领颇颈颉颊颋颌颍颎颏颐频颓颔颖颗题颙颚颛颜额'
    '颞颟颠颡颢颤颥颦颧风飏飐飑飒飓飔飕飗飘飙飞食飧飨餍餐餮饔饕饥饧饨饩饪饫饬饭饮饯饰'
    '饱饲饳饴饵饶饷饸饹饺饻饼饽饿馁馃馄馅馆馇馈馉馊馋馌馍馏馐馑馒馓馔馕首馗馘香馝馞馥'
    '馧馨马驭驮驯驰驱驲驳驴驵驶驷驸驹驺驻驼驽驾驿骀骁骂骃骄骅骆骇骈骉骊骋验骍骎骏骐骑'
    '骒骓骕骖骗骘骙骚骛骜骝骞骟骠骡骢骣骤骥骦骧骨骰骱骶骷骸骺骼髀髁髂髃髅髋髌髎髑髓高'
    '髡髢髦髫髭髯髹髻髽鬃鬈鬏鬒鬓鬘鬟鬣鬯鬲鬶鬷鬻鬼魁魂魃魄魅魆魇魈魉魋魍魏魑魔鱼鱽鱾'
    '鱿鲀鲁鲂鲃鲅鲆鲇鲈鲉鲊鲋鲌鲍鲎鲏鲐鲑鲒鲔鲕鲖鲗鲘鲙鲚鲛鲜鲝鲞鲟鲠鲡鲢鲣鲤鲥鲦鲧鲨'
    '鲩鲪鲫鲬鲭鲮鲯鲰鲱鲲鲳鲴鲵鲷鲸鲹鲺鲻鲼鲽鲾鲿鳀鳁鳂鳃鳄鳅鳇鳈鳉鳊鳌鳍鳎鳏鳐鳑鳒鳓'
    '鳔鳕鳖鳗鳘鳙鳚鳛鳜鳝鳞鳟鳠鳡鳢鳣鳤鸟鸠鸡鸢鸣鸤鸥鸦鸧鸨鸩鸪鸫鸬鸭鸮鸯鸰鸱鸲鸳鸵鸶'
    '鸷鸸鸹鸺鸻鸼鸽鸾鸿鹀鹁鹂鹃鹄鹅鹆鹇鹈鹉鹊鹋鹌鹍鹎鹏鹐鹑鹒鹔鹕鹖鹗鹘鹙鹚鹛鹜鹝鹞鹟'
    '鹠鹡鹢鹣鹤鹦鹧鹨鹩鹪鹫鹬鹭鹮鹯鹰鹱鹲鹳鹴鹾鹿麀麂麇麈麋麑麒麓麖麝麟麦麸麹麻麽麾黄'
    '黇黉黍黎黏黑黔默黛黜黝黟黠黡黢黥黧黩黪黯黹黻黼黾鼋鼍鼎鼐鼒鼓鼗鼙鼠鼢鼩鼫鼬鼯鼱鼷'
    '鼹鼻鼽鼾齁齇齉齐齑齿龀龁龂龃龄龅龆龇龈龉龊龋龌龙龚龛龟龠龢鿍鿎鿏㑇㑊㕮㘎㙍㙘㙦㛃'
    '㛚㛹㟃㠇㠓㤘㥄㧐㧑㧟㫰㬊㬎㬚㭎㭕㮾㰀㳇㳘㳚㴔㵐㶲㸆㸌㺄㻬㽏㿠䁖䂮䃅䃎䅟䌹䎃䎖䏝䏡'
    '䏲䐃䓖䓛䓨䓫䓬䗖䗛䗪䗴䜣䝙䢺䢼䣘䥽䦃䲟䲠䲢䴓䴔䴕䴖䴗䴘䴙䶮𠅤𠙶𠳐𡎚𡐓𣗋𣲗𣲘𣸣𤧛𤩽'
    '𤫉𥔲𥕢𥖨𥻗𦈡𦒍𦙶𦝼𦭜𦰡𧿹𨐈𨙸𨚕𨟠𨭉𨱇𨱏𨱑𨱔𨺙𩽾𩾃𩾌𪟝𪣻𪤗𪨰𪨶𪩘𪾢𫄧𫄨𫄷𫄸𫇭𫌀𫍣𫍯'
    '𫍲𫍽𫐄𫐐𫐓𫑡𫓧𫓯𫓶𫓹𫔍𫔎𫔶𫖮𫖯𫖳𫗧𫗴𫘜𫘝𫘦𫘧𫘨𫘪𫘬𫚕𫚖𫚭𫛭𫞩𫟅𫟦𫟹𫟼𫠆𫠊𫠜𫢸𫫇𫭟'
    '𫭢𫭼𫮃𫰛𫵷𫶇𫷷𫸩𬀩𬀪𬂩𬃊𬇕𬇙𬇹𬉼𬊈𬊤𬌗𬍛𬍡𬍤𬒈𬒔𬒗𬕂𬘓𬘘𬘡𬘩𬘫𬘬𬘭𬘯𬙂𬙊𬙋𬜬𬜯𬞟'
    '𬟁𬟽𬣙𬣞𬣡𬣳𬤇𬤊𬤝𬨂𬨎𬩽𬪩𬬩𬬭𬬮𬬱𬬸𬬹𬬻𬬿𬭁𬭊𬭎𬭚𬭛𬭤𬭩𬭬𬭯𬭳𬭶𬭸𬭼𬮱𬮿𬯀𬯎𬱖𬱟'
    '𬳵𬳶𬳽𬳿𬴂𬴃𬴊𬶋𬶍𬶏𬶐𬶟𬶠𬶨𬶭𬶮𬷕𬸘𬸚𬸣𬸦𬸪𬹼𬺈𬺓'
)
CN_CHARS_EXT = '吶诶屌囧飚屄'  # 扩展中文字符集（网络用语等）

CN_CHARS = CN_CHARS_COMMON + CN_CHARS_EXT  # 完整中文字符集
# 中文字符快速查找字典
IN_CH_CHARS = {c: True for c in CN_CHARS}

# 英文字母和数字字符集
EN_CHARS = string.ascii_letters + string.digits
# 英文字符快速查找字典
IN_EN_CHARS = {c: True for c in EN_CHARS}

# 有效字符集（中文 + 英文 + 空格 + 标点）
VALID_CHARS = CN_CHARS + EN_CHARS + ' ' + PUNCS
# 有效字符快速查找字典
IN_VALID_CHARS = {c: True for c in VALID_CHARS}


# ============================================================================ #
# 基础类定义
# 中文数字系统的核心抽象类
# ============================================================================ #

class ChineseChar(object):
    """
    中文字符基类

    每个 ChineseChar 对象对应一个中文字符，同时包含简体字和繁体字两种形式。
    在转换时可以根据需要输出简体或繁体的字符串表示。

    属性:
        simplified (str): 简体中文字符
        traditional (str): 繁体中文字符
    """

    def __init__(self, simplified, traditional):
        """
        初始化 ChineseChar 实例

        参数:
            simplified (str): 简体字
            traditional (str): 繁体字
        """
        self.simplified = simplified
        self.traditional = traditional

    def __str__(self):
        """
        返回字符串表示（优先返回简体，其次繁体）

        返回:
            str: 中文字符或 None
        """
        return self.simplified or self.traditional or None

    def __repr__(self):
        """返回正式的字符串表示（与 __str__ 相同）"""
        return self.__str__()


class ChineseNumberUnit(ChineseChar):
    """
    中文数位/单位字符类（如：十、百、千、万、亿等）

    继承自 ChineseChar，除了简体/繁体外，还包含幂指数和大写形式。
    power 属性表示该数位对应的 10 的幂指数。

    属性:
        power (int): 数位的幂指数，如 十=1, 百=2, 千=3, 万=4
        big_s (str): 简体大写形式（如：拾、佰、仟）
        big_t (str): 繁体大写形式（如：拾、佰、仟）
    """

    def __init__(self, power, simplified, traditional, big_s, big_t):
        """
        初始化 ChineseNumberUnit 实例

        参数:
            power (int): 10 的幂指数
            simplified (str): 简体字
            traditional (str): 繁体字
            big_s (str): 简体大写形式
            big_t (str): 繁体大写形式
        """
        super(ChineseNumberUnit, self).__init__(simplified, traditional)
        self.power = power
        self.big_s = big_s
        self.big_t = big_t

    def __str__(self):
        """返回 10^power 形式的字符串表示"""
        return '10^{}'.format(self.power)

    @classmethod
    def create(cls, index, value, numbering_type=NUMBERING_TYPES[1], small_unit=False):
        """
        工厂方法：根据索引和数字系统类型创建数位实例

        不同的数字系统类型（low/mid/high）决定了大数位的幂指数计算方式：
          - low:  '兆' = '亿' * '十' = 1e9
          - mid:  '兆' = '亿' * '万' = 1e12
          - high: '兆' = '亿' * '亿' = 1e16

        参数:
            index (int): 在数位列表中的索引
            value (tuple): 包含 (简体, 繁体) 的字符串元组
            numbering_type (str): 数字系统类型，'low'、'mid' 或 'high'
            small_unit (bool): 是否为小单位（十、百、千、万）

        返回:
            ChineseNumberUnit: 创建好的数位实例
        """
        if small_unit:
            # 小单位：十、百、千、万，幂指数为 index + 1
            return ChineseNumberUnit(power=index + 1,
                                     simplified=value[0], traditional=value[1], big_s=value[1], big_t=value[1])
        elif numbering_type == NUMBERING_TYPES[0]:
            # low 类型：幂指数 = index + 8
            return ChineseNumberUnit(power=index + 8,
                                     simplified=value[0], traditional=value[1], big_s=value[0], big_t=value[1])
        elif numbering_type == NUMBERING_TYPES[1]:
            # mid 类型：幂指数 = (index + 2) * 4
            return ChineseNumberUnit(power=(index + 2) * 4,
                                     simplified=value[0], traditional=value[1], big_s=value[0], big_t=value[1])
        elif numbering_type == NUMBERING_TYPES[2]:
            # high 类型：幂指数 = 2^(index + 3)
            return ChineseNumberUnit(power=pow(2, index + 3),
                                     simplified=value[0], traditional=value[1], big_s=value[0], big_t=value[1])
        else:
            raise ValueError(
                'Counting type should be in {0} ({1} provided).'.format(NUMBERING_TYPES, numbering_type))


class ChineseNumberDigit(ChineseChar):
    """
    中文数字字符类（如：零、一、二、三、四等）

    继承自 ChineseChar，包含数字的数值、大写形式和替代写法。
    替代写法用于特殊场景，如：零可写作"〇"（年份），一可写作"幺"（电话号码），
    二可写作"两"（量词前）。

    属性:
        value (int): 数字的数值（0-9）
        big_s (str): 简体大写形式
        big_t (str): 繁体大写形式
        alt_s (str): 简体替代写法
        alt_t (str): 繁体替代写法
    """

    def __init__(self, value, simplified, traditional, big_s, big_t, alt_s=None, alt_t=None):
        """
        初始化 ChineseNumberDigit 实例

        参数:
            value (int): 数字值
            simplified (str): 简体字
            traditional (str): 繁体字
            big_s (str): 简体大写形式
            big_t (str): 繁体大写形式
            alt_s (str, optional): 简体替代写法
            alt_t (str, optional): 繁体替代写法
        """
        super(ChineseNumberDigit, self).__init__(simplified, traditional)
        self.value = value
        self.big_s = big_s
        self.big_t = big_t
        self.alt_s = alt_s
        self.alt_t = alt_t

    def __str__(self):
        """返回数字值的字符串表示"""
        return str(self.value)

    @classmethod
    def create(cls, i, v):
        """
        工厂方法：创建中文数字字符实例

        参数:
            i (int): 数字值（0-9）
            v (tuple): 包含 (简体, 繁体, 大写简体, 大写繁体) 的元组

        返回:
            ChineseNumberDigit: 创建好的数字字符实例
        """
        return ChineseNumberDigit(i, v[0], v[1], v[2], v[3])


class ChineseMath(ChineseChar):
    """
    中文数学符号类（如：正、负、点）

    用于表示数字系统中的数学符号，包含符号标识和计算表达式。

    属性:
        symbol (str): 数学符号（如 '+', '-', '.')
        expression (function): 对应的计算函数
    """

    def __init__(self, simplified, traditional, symbol, expression=None):
        """
        初始化 ChineseMath 实例

        参数:
            simplified (str): 简体字
            traditional (str): 繁体字
            symbol (str): 数学符号
            expression (function, optional): 计算函数
        """
        super(ChineseMath, self).__init__(simplified, traditional)
        self.symbol = symbol
        self.expression = expression
        # 数学符号的大写形式就是其本身
        self.big_s = simplified
        self.big_t = traditional


# 类别名：简化类名引用
CC, CNU, CND, CM = ChineseChar, ChineseNumberUnit, ChineseNumberDigit, ChineseMath


class NumberSystem(object):
    """
    中文数字系统类

    包含所有数位单位、数字字符和数学符号的完整数字系统。
    不同的数字系统类型（low/mid/high）影响大数位的幂指数计算。
    """
    pass


class MathSymbol(object):
    """
    数学符号容器类

    用于组织中文数字系统中的数学符号（正、负、小数点）。

    属性:
        positive (ChineseMath): 正号
        negative (ChineseMath): 负号
        point (ChineseMath): 小数点
    """

    def __init__(self, positive, negative, point):
        """
        初始化 MathSymbol 实例

        参数:
            positive (ChineseMath): 正号对象
            negative (ChineseMath): 负号对象
            point (ChineseMath): 小数点对象
        """
        self.positive = positive
        self.negative = negative
        self.point = point

    def __iter__(self):
        """迭代所有数学符号"""
        for v in self.__dict__.values():
            yield v


# ============================================================================ #
# 核心工具函数：数字系统创建与数字转换
# ============================================================================ #

def create_system(numbering_type=NUMBERING_TYPES[1]):
    """
    根据数字系统类型创建完整的数字系统

    构建包括所有数位单位、数字字符和数学符号的完整数字系统。
    不同类型（low/mid/high）影响大数位（亿以上）的幂指数计算方式。

    参数:
        numbering_type (str): 数字系统类型，默认为 'mid'
            NUMBERING_TYPES = ['low', 'mid', 'high']
            - low:  '兆' = '亿' * '十' = 10^9,  '京' = '兆' * '十', 以此类推
            - mid:  '兆' = '亿' * '万' = 10^12, '京' = '兆' * '万', 以此类推
            - high: '兆' = '亿' * '亿' = 10^16, '京' = '兆' * '兆', 以此类推

    返回:
        NumberSystem: 完整的数字系统对象
    """
    # 构建大数位单位（亿、兆、京、垓等）
    # 将简体和繁体版本合并成元组列表
    all_larger_units = zip(
        LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED, LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL)
    larger_units = [CNU.create(i, v, numbering_type, False)
                    for i, v in enumerate(all_larger_units)]

    # 构建小数位单位（十、百、千、万）
    all_smaller_units = zip(
        SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED, SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL)
    smaller_units = [CNU.create(i, v, small_unit=True)
                     for i, v in enumerate(all_smaller_units)]

    # 构建数字字符（零到九）
    chinese_digis = zip(CHINESE_DIGIS, CHINESE_DIGIS,
                        BIG_CHINESE_DIGIS_SIMPLIFIED, BIG_CHINESE_DIGIS_TRADITIONAL)
    digits = [CND.create(i, v) for i, v in enumerate(chinese_digis)]
    # 设置特殊替代写法
    digits[0].alt_s, digits[0].alt_t = ZERO_ALT, ZERO_ALT      # 零 -> 〇
    digits[1].alt_s, digits[1].alt_t = ONE_ALT, ONE_ALT        # 一 -> 幺
    digits[2].alt_s, digits[2].alt_t = TWO_ALTS[0], TWO_ALTS[1]  # 二 -> 两/兩

    # 构建数学符号
    positive_cn = CM(POSITIVE[0], POSITIVE[1], '+', lambda x: x)               # 正号
    negative_cn = CM(NEGATIVE[0], NEGATIVE[1], '-', lambda x: -x)              # 负号
    point_cn = CM(POINT[0], POINT[1], '.', lambda x, y: float(str(x) + '.' + str(y)))  # 小数点

    # 组装完整的数字系统
    system = NumberSystem()
    system.units = smaller_units + larger_units    # 所有数位单位
    system.digits = digits                          # 所有数字字符
    system.math = MathSymbol(positive_cn, negative_cn, point_cn)  # 数学符号

    return system


def chn2num(chinese_string, numbering_type=NUMBERING_TYPES[1]):
    """
    将中文数字字符串转换为阿拉伯数字

    例如："一万二千三百四十五" -> "12345"
          "三点一四" -> "3.14"
          "负二十" -> "-20"

    参数:
        chinese_string (str): 中文数字字符串
        numbering_type (str): 数字系统类型，默认为 'mid'

    返回:
        str: 转换后的阿拉伯数字字符串
    """

    def get_symbol(char, system):
        """
        在数字系统中查找字符对应的符号对象

        参数:
            char (str): 要查找的中文字符
            system (NumberSystem): 数字系统

        返回:
            ChineseNumberUnit|ChineseNumberDigit|ChineseMath: 对应的符号对象，未找到则返回 None
        """
        # 先在数位单位中查找
        for u in system.units:
            if char in [u.traditional, u.simplified, u.big_s, u.big_t]:
                return u
        # 再在数字字符中查找（含替代写法）
        for d in system.digits:
            if char in [d.traditional, d.simplified, d.big_s, d.big_t, d.alt_s, d.alt_t]:
                return d
        # 最后在数学符号中查找
        for m in system.math:
            if char in [m.traditional, m.simplified]:
                return m

    def string2symbols(chinese_string, system):
        """
        将中文数字字符串拆分为符号对象列表

        分别处理整数部分和小数部分。

        参数:
            chinese_string (str): 中文数字字符串
            system (NumberSystem): 数字系统

        返回:
            tuple: (整数部分符号列表, 小数部分符号列表)
        """
        int_string, dec_string = chinese_string, ''
        # 检查是否有小数点
        for p in [system.math.point.simplified, system.math.point.traditional]:
            if p in chinese_string:
                int_string, dec_string = chinese_string.split(p)
                break
        # 将字符串转换为符号对象列表
        return [get_symbol(c, system) for c in int_string], \
               [get_symbol(c, system) for c in dec_string]

    def correct_symbols(integer_symbols, system):
        """
        修正整数部分的符号序列

        处理以下特殊情况：
        1. "一百八" -> "一百八十"（末尾缺少零位单位）
        2. "一亿一千三百万" -> "一亿 一千万 三百万"（合并重叠单位）

        参数:
            integer_symbols (list): 整数部分符号对象列表
            system (NumberSystem): 数字系统

        返回:
            list: 修正后的符号对象列表
        """
        # 如果以数位单位开头（如"十二"），在前面补上"一"
        if integer_symbols and isinstance(integer_symbols[0], CNU):
            if integer_symbols[0].power == 1:
                integer_symbols = [system.digits[1]] + integer_symbols

        # 处理末尾缺少单位的情况：如果最后两个符号是"数字+单位"，则在末尾追加低一级的单位
        # 例如："一百八"中"八"是数字，"百"是单位，需要追加"十"
        if len(integer_symbols) > 1:
            if isinstance(integer_symbols[-1], CND) and isinstance(integer_symbols[-2], CNU):
                integer_symbols.append(
                    CNU(integer_symbols[-2].power - 1, None, None, None, None))

        # 合并重叠的单位（如："三千"+"百万"->"三千万"）
        result = []
        unit_count = 0
        for s in integer_symbols:
            if isinstance(s, CND):
                result.append(s)
                unit_count = 0
            elif isinstance(s, CNU):
                current_unit = CNU(s.power, None, None, None, None)
                unit_count += 1

            if unit_count == 1:
                result.append(current_unit)
            elif unit_count > 1:
                # 合并单位：将前一个较小单位的值加上当前单位的值
                for i in range(len(result)):
                    if isinstance(result[-i - 1], CNU) and result[-i - 1].power < current_unit.power:
                        result[-i - 1] = CNU(result[-i - 1].power +
                                             current_unit.power, None, None, None, None)
        return result

    def compute_value(integer_symbols):
        """
        计算整数部分的数值

        当当前单位大于前一个单位时，当前单位会乘以前面所有单位的值。
        例如："两千万" = 2 * 1000 * 10000 = 20000000，而非 2000 + 10000

        参数:
            integer_symbols (list): 修正后的整数部分符号列表

        返回:
            int: 计算得到的整数值
        """
        value = [0]
        last_power = 0
        for s in integer_symbols:
            if isinstance(s, CND):
                # 数字符号：设置当前段的值
                value[-1] = s.value
            elif isinstance(s, CNU):
                # 单位符号：将当前段的值乘以 10^power
                value[-1] *= pow(10, s.power)
                if s.power > last_power:
                    # 当前单位大于上一个单位时，之前所有段都乘以这个单位
                    value[:-1] = list(map(lambda v: v *
                                          pow(10, s.power), value[:-1]))
                    last_power = s.power
                value.append(0)  # 开始新段
        return sum(value)

    # 主流程
    system = create_system(numbering_type)
    int_part, dec_part = string2symbols(chinese_string, system)
    int_part = correct_symbols(int_part, system)
    int_str = str(compute_value(int_part))
    dec_str = ''.join([str(d.value) for d in dec_part])
    # 如果有小数部分，拼接整数和小数
    if dec_part:
        return '{0}.{1}'.format(int_str, dec_str)
    else:
        return int_str


def num2chn(number_string, numbering_type=NUMBERING_TYPES[1], big=False,
            traditional=False, alt_zero=False, alt_one=False, alt_two=True,
            use_zeros=True, use_units=True):
    """
    将阿拉伯数字字符串转换为中文数字

    例如："12345" -> "一万二千三百四十五"
          "3.14" -> "三点一四"

    参数:
        number_string (str): 阿拉伯数字字符串
        numbering_type (str): 数字系统类型，默认 'mid'
        big (bool): 是否使用大写形式（财务用），默认 False
        traditional (bool): 是否使用繁体字，默认 False
        alt_zero (bool): 是否使用"〇"替代"零"，默认 False
        alt_one (bool): 是否使用"幺"替代"一"，默认 False
        alt_two (bool): 是否使用"两"替代"二"，默认 True
        use_zeros (bool): 是否在中间补零，默认 True
        use_units (bool): 是否使用数位单位，默认 True

    返回:
        str: 转换后的中文数字字符串
    """

    def get_value(value_string, use_zeros=True):
        """
        递归构建数字的值表示

        将数字字符串按位分解为数位和数字符号的序列。
        例如："12345" -> [一, 万, 二千, 三百, 四十, 五]

        参数:
            value_string (str): 数字字符串
            use_zeros (bool): 是否在中间位置补零

        返回:
            list: 符号对象列表
        """
        # 去除前导零
        striped_string = value_string.lstrip('0')

        # 如果全部是零，返回空列表
        if not striped_string:
            return []

        # 如果只有一位数字
        elif len(striped_string) == 1:
            if use_zeros and len(value_string) != len(striped_string):
                # 有前导零需要补"零"
                return [system.digits[0], system.digits[int(striped_string)]]
            else:
                return [system.digits[int(striped_string)]]

        # 递归处理多位数字
        else:
            # 找到小于数字位数的最大的数位单位
            result_unit = next(u for u in reversed(
                system.units) if u.power < len(striped_string))
            # 分割为高位部分和低位部分
            result_string = value_string[:-result_unit.power]
            return get_value(result_string) + [result_unit] + get_value(striped_string[-result_unit.power:])

    # 创建数字系统
    system = create_system(numbering_type)

    # 分离整数部分和小数部分
    int_dec = number_string.split('.')
    if len(int_dec) == 1:
        int_string = int_dec[0]
        dec_string = ""
    elif len(int_dec) == 2:
        int_string = int_dec[0]
        dec_string = int_dec[1]
    else:
        raise ValueError(
            "invalid input num string with more than one dot: {}".format(number_string))

    # 生成整数部分的符号序列
    if use_units and len(int_string) > 1:
        result_symbols = get_value(int_string)
    else:
        result_symbols = [system.digits[int(c)] for c in int_string]

    # 生成小数部分的符号序列
    dec_symbols = [system.digits[int(c)] for c in dec_string]
    if dec_string:
        result_symbols += [system.math.point] + dec_symbols

    # 智能使用"两"替代"二"
    if alt_two:
        liang = CND(2, system.digits[2].alt_s, system.digits[2].alt_t,
                    system.digits[2].big_s, system.digits[2].big_t)
        for i, v in enumerate(result_symbols):
            if isinstance(v, CND) and v.value == 2:
                next_symbol = result_symbols[i +
                                             1] if i < len(result_symbols) - 1 else None
                previous_symbol = result_symbols[i - 1] if i > 0 else None
                # 在量词前的"二"改为"两"（如"两个"而非"二个"）
                if isinstance(next_symbol, CNU) and isinstance(previous_symbol, (CNU, type(None))):
                    if next_symbol.power != 1 and ((previous_symbol is None) or (previous_symbol.power != 1)):
                        result_symbols[i] = liang

    # 选择输出属性（简体/繁体、普通/大写）
    if big:
        attr_name = 'big_'
        if traditional:
            attr_name += 't'
        else:
            attr_name += 's'
    else:
        if traditional:
            attr_name = 'traditional'
        else:
            attr_name = 'simplified'

    # 将符号序列转换为字符串
    result = ''.join([getattr(s, attr_name) for s in result_symbols])

    # 可选地使用替代零写法
    if alt_zero:
        result = result.replace(
            getattr(system.digits[0], attr_name), system.digits[0].alt_s)

    # 可选地使用替代一写法
    if alt_one:
        result = result.replace(
            getattr(system.digits[1], attr_name), system.digits[1].alt_s)

    # 处理以"点"开头的情况（如".5" -> "零点五"）
    for i, p in enumerate(POINT):
        if result.startswith(p):
            return CHINESE_DIGIS[0] + result

    # 处理"一十"到"十九"的简写（去掉前面的"一"）
    # 如"一十五" -> "十五"
    if len(result) >= 2 and result[1] in [SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED[0],
                                          SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL[0]] and \
            result[0] in [CHINESE_DIGIS[1], BIG_CHINESE_DIGIS_SIMPLIFIED[1], BIG_CHINESE_DIGIS_TRADITIONAL[1]]:
        result = result[1:]

    return result


# ============================================================================ #
# NSW（Non-Standard Word）规范化类
# 每个类对应一种非标准写法的转换规则
# ============================================================================ #

class Cardinal:
    """
    基数词转换类（CARDINAL）

    用于普通数字的转换，如 "123" -> "一百二十三"
    """

    def __init__(self, cardinal=None, chntext=None):
        """
        初始化 Cardinal 实例

        参数:
            cardinal (str): 阿拉伯数字字符串
            chntext (str): 中文数字字符串
        """
        self.cardinal = cardinal
        self.chntext = chntext

    def chntext2cardinal(self):
        """中文数字转阿拉伯数字：调用 chn2num"""
        return chn2num(self.chntext)

    def cardinal2chntext(self):
        """阿拉伯数字转中文数字：调用 num2chn"""
        return num2chn(self.cardinal)


class Digit:
    """
    数字编号转换类（DIGIT）

    用于将长数字逐位读出（如电话号码、编号等），而非按数值转换。
    例如："1234" -> "一二三四"（而非"一千二百三十四"）
    """

    def __init__(self, digit=None, chntext=None):
        """
        初始化 Digit 实例

        参数:
            digit (str): 数字字符串
            chntext (str): 中文逐位数字字符串
        """
        self.digit = digit
        self.chntext = chntext

    def digit2chntext(self):
        """
        将数字逐位转换为中文（不使用单位，不使用"两"）

        例如："1234" -> "一二三四"

        返回:
            str: 逐位转换的中文数字
        """
        return num2chn(self.digit, alt_two=False, use_units=False)


class TelePhone:
    """
    电话号码转换类（TELEPHONE）

    将电话号码转换为中文读法，支持手机号和固定电话。
    例如：
      手机号："13800138000" -> "幺三八零零幺三八零零零"
      固话："010-12345678" -> "零幺零幺二三四五六七八"
    """

    def __init__(self, telephone=None, raw_chntext=None, chntext=None):
        """
        初始化 TelePhone 实例

        参数:
            telephone (str): 电话号码字符串
            raw_chntext (str): 包含间隔标记的原始中文文本
            chntext (str): 最终中文文本
        """
        self.telephone = telephone
        self.raw_chntext = raw_chntext
        self.chntext = chntext

    def telephone2chntext(self, fixed=False):
        """
        将电话号码转换为中文读法

        参数:
            fixed (bool): 是否为固定电话（使用"-"分割，无间隔直接拼接）
                          False 为手机号（使用空格分割，无间隔直接拼接）

        返回:
            str: 中文电话号码
        """
        if fixed:
            # 固定电话：按"-"分割后逐段转换，直接拼接
            sil_parts = self.telephone.split('-')
            self.raw_chntext = '<SIL>'.join([
                num2chn(part, alt_two=False, use_units=False) for part in sil_parts
            ])
            self.chntext = self.raw_chntext.replace('<SIL>', '')
        else:
            # 手机号：按空格分割（如果有）后逐段转换，直接拼接
            sp_parts = self.telephone.strip('+').split()
            self.raw_chntext = '<SP>'.join([
                num2chn(part, alt_two=False, use_units=False) for part in sp_parts
            ])
            self.chntext = self.raw_chntext.replace('<SP>', '')
        return self.chntext


class Fraction:
    """
    分数转换类（FRACTION）

    将分数转换为中文读法。
    例如："3/4" -> "四分之三"
    """

    def __init__(self, fraction=None, chntext=None):
        """
        初始化 Fraction 实例

        参数:
            fraction (str): 分数字符串（如 "3/4"）
            chntext (str): 中文分数文本
        """
        self.fraction = fraction
        self.chntext = chntext

    def chntext2fraction(self):
        """中文分数转阿拉伯分数"""
        denominator, numerator = self.chntext.split('分之')
        return chn2num(numerator) + '/' + chn2num(denominator)

    def fraction2chntext(self):
        """
        阿拉伯分数转中文

        返回:
            str: 中文分数，如 "四分之三"
        """
        numerator, denominator = self.fraction.split('/')
        return num2chn(denominator) + '分之' + num2chn(numerator)


class Date:
    """
    日期转换类（DATE）

    将日期字符串转换为中文读法。
    例如："2024年1月15日" -> "二零二四年一月十五日"
    """

    def __init__(self, date=None, chntext=None):
        """
        初始化 Date 实例

        参数:
            date (str): 日期字符串
            chntext (str): 中文日期文本
        """
        self.date = date
        self.chntext = chntext

    def date2chntext(self):
        """
        将日期转换为中文读法

        年份逐位转换（如"2024"->"二零二四"），月份和日期使用基数转换。
        支持以下格式：
          - 完整格式："2024年1月15日"
          - 只带月日："1月15日"
          - 只带日："15日"

        返回:
            str: 中文日期文本
        """
        date = self.date
        try:
            # 提取年份："2024年1月15日" -> year="2024", other="1月15日"
            year, other = date.strip().split('年', 1)
            year = Digit(digit=year).digit2chntext() + '年'  # 年份逐位转换
        except ValueError:
            # 没有年份部分
            other = date
            year = ''
        if other:
            try:
                # 提取月份："1月15日" -> month="1", day="15日"
                month, day = other.strip().split('月', 1)
                month = Cardinal(cardinal=month).cardinal2chntext() + '月'
            except ValueError:
                # 没有月份部分
                day = date
                month = ''
            if day:
                # 去除末尾的"日"或"号"，转换数字部分后再添上
                day = Cardinal(cardinal=day[:-1]).cardinal2chntext() + day[-1]
        else:
            month = ''
            day = ''
        chntext = year + month + day
        self.chntext = chntext
        return self.chntext


class Money:
    """
    货币金额转换类（MONEY）

    将货币金额转换为中文读法。
    例如："123.45元" -> "一百二十三点四五元"
    """

    def __init__(self, money=None, chntext=None):
        """
        初始化 Money 实例

        参数:
            money (str): 货币金额字符串
            chntext (str): 中文金额文本
        """
        self.money = money
        self.chntext = chntext

    def money2chntext(self):
        """
        将货币金额中的数字部分转换为中文

        返回:
            str: 中文金额文本
        """
        money = self.money
        # 匹配金额中的数字部分（包括小数）
        pattern = re.compile(r'(\d+(\.\d+)?)')
        matchers = pattern.findall(money)
        if matchers:
            for matcher in matchers:
                # 用中文数字替换阿拉伯数字
                money = money.replace(matcher[0], Cardinal(
                    cardinal=matcher[0]).cardinal2chntext())
        self.chntext = money
        return self.chntext


class Percentage:
    """
    百分数转换类（PERCENTAGE）

    将百分数转换为中文读法。
    例如："50%" -> "百分之五十"
    """

    def __init__(self, percentage=None, chntext=None):
        """
        初始化 Percentage 实例

        参数:
            percentage (str): 百分数字符串
            chntext (str): 中文百分数文本
        """
        self.percentage = percentage
        self.chntext = chntext

    def chntext2percentage(self):
        """中文百分数转阿拉伯百分数"""
        return chn2num(self.chntext.strip().strip('百分之')) + '%'

    def percentage2chntext(self):
        """
        阿拉伯百分数转中文

        返回:
            str: 中文百分数，如 "百分之五十"
        """
        return '百分之' + num2chn(self.percentage.strip().strip('%'))


# ============================================================================ #
# NSW 规范化主函数
# 按顺序应用各种规范化规则
# ============================================================================ #

def normalize_nsw(raw_text):
    """
    NSW（Non-Standard Word）规范化主函数

    将文本中的非标准写法按以下顺序转换为标准中文读音格式：
      1. 日期规范化
      2. 货币金额规范化
      3. 电话号码规范化（手机号 + 固话）
      4. 分数规范化
      5. 百分数规范化
      6. 基数词+量词规范化
      7. 长数字编号规范化
      8. 一般基数词规范化
      9. 特殊模式恢复（如 P2P, O2O 等）

    参数:
        raw_text (str): 原始文本

    返回:
        str: 规范化后的文本
    """
    # 在文本前后添加标记，方便正则匹配边界
    text = '^' + raw_text + '$'

    # ---- 1. 规范化日期 ----
    # 匹配格式：可选年份（以 0/8/9 或 19/20 开头）+ 月 + 日
    pattern = re.compile(
        r"\D+((([089]\d|(19|20)\d{2})年)?(\d{1,2}月(\d{1,2}[日号])?)?)")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher[0], Date(
                date=matcher[0]).date2chntext(), 1)

    # ---- 2. 规范化货币金额 ----
    # 匹配格式：数字 + 货币单位（元/块/角/分等）
    pattern = re.compile(
        r"\D+((\d+(\.\d+)?)[多余几]?" + CURRENCY_UNITS + r"(\d" + CURRENCY_UNITS + r"?)?)")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher[0], Money(
                money=matcher[0]).money2chntext(), 1)

    # ---- 3. 规范化电话号码 ----
    # 3.1 手机号码
    # 匹配规则：+86 可选 + 1开头 + 第二位 3/8/5/7/9 等 + 8位数字
    pattern = re.compile(
        r"\D((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})\D")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher[0], TelePhone(
                telephone=matcher[0]).telephone2chntext(), 1)

    # 3.2 固定电话
    # 匹配规则：区号（可选）+ 7-8位号码
    pattern = re.compile(r"\D((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})\D")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher[0], TelePhone(
                telephone=matcher[0]).telephone2chntext(fixed=True), 1)

    # ---- 4. 规范化分数 ----
    # 匹配格式：数字/数字
    pattern = re.compile(r"(\d+/\d+)")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher, Fraction(
                fraction=matcher).fraction2chntext(), 1)

    # ---- 5. 规范化百分数 ----
    # 先将全角百分号转换为半角
    text = text.replace('％', '%')
    pattern = re.compile(r"(\d+(\.\d+)?%)")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher[0], Percentage(
                percentage=matcher[0]).percentage2chntext(), 1)

    # ---- 6. 规范化基数词+量词 ----
    # 匹配格式：数字 + 量词（如"3个"、"5斤"）
    pattern = re.compile(r"(\d+(\.\d+)?)[多余几]?" + COM_QUANTIFIERS)
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher[0], Cardinal(
                cardinal=matcher[0]).cardinal2chntext(), 1)

    # ---- 7. 规范化长数字编号（4位以上） ----
    # 长数字通常按位读出，而非按数值转换
    pattern = re.compile(r"(\d{4,32})")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher, Digit(
                digit=matcher).digit2chntext(), 1)

    # ---- 8. 规范化一般基数词 ----
    # 匹配剩余的所有数字（含小数）
    pattern = re.compile(r"(\d+(\.\d+)?)")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher[0], Cardinal(
                cardinal=matcher[0]).cardinal2chntext(), 1)

    # ---- 9. 特殊模式恢复 ----
    # 将 A二B 模式（如"P二P"）中的"二"恢复为"2"
    # 这是为了处理前面数字转换时被误转的情况
    pattern = re.compile(r"(([a-zA-Z]+)二([a-zA-Z]+))")
    matchers = pattern.findall(text)
    if matchers:
        for matcher in matchers:
            text = text.replace(matcher[0], matcher[1]+'2'+matcher[2], 1)

    # 去除前后添加的边界标记并返回
    return text.lstrip('^').rstrip('$')


# ============================================================================ #
# 文本清理辅助函数
# ============================================================================ #

def remove_erhua(text):
    """
    去除文本中的儿化音，但保留白名单中的合法"儿"字

    儿化音在文字中写作"儿"，但在朗读时不应该读出。
    例如："他女儿在那边儿" -> "他女儿在那边"
    白名单中的词（如"儿童"、"儿子"等）不受影响。

    参数:
        text (str): 输入文本

    返回:
        str: 去除儿化音后的文本
    """
    new_str = ''
    # 循环查找文本中的"儿"字
    while re.search('儿', text):
        a = re.search('儿', text).span()  # 找到"儿"的位置
        remove_er_flag = 0  # 标记是否应该去除

        # 检查该"儿"是否在白名单词中
        if ER_WHITELIST_PATTERN.search(text):
            b = ER_WHITELIST_PATTERN.search(text).span()
            # 如果白名单词的起始位置 <= "儿"的位置，则保留
            if b[0] <= a[0]:
                remove_er_flag = 1

        if remove_er_flag == 0:
            # 不在白名单中，去除"儿"
            new_str = new_str + text[0:a[0]]
            text = text[a[1]:]
        else:
            # 在白名单中，保留整个词
            new_str = new_str + text[0:b[1]]
            text = text[b[1]:]

    # 拼接剩余文本
    text = new_str + text
    return text


def remove_space(text):
    """
    智能移除空格但保留英文单词之间的空格

    英文单词之间需要保留空格，但中文与中文之间不需要空格。
    函数在英文/数字字符之间保留空格，其他情况移除。

    参数:
        text (str): 输入文本

    返回:
        str: 处理后的文本
    """
    tokens = text.split()
    new = []
    for k, t in enumerate(tokens):
        if k != 0:
            # 如果前一个 token 的最后一个字符和后一个 token 的第一个字符都是英文/数字，
            # 则保留空格
            if IN_EN_CHARS.get(tokens[k-1][-1]) and IN_EN_CHARS.get(t[0]):
                new.append(' ')
        new.append(t)
    return ''.join(new)


# ============================================================================ #
# 主规范化类：TextNorm
# 提供统一的文本规范化接口，可通过参数配置各种规范化选项
# ============================================================================ #

class TextNorm:
    """
    中文文本规范化器

    将原始文本进行一系列规范化处理，使其适合 TTS（文本转语音）系统使用。
    支持多种处理选项的灵活组合。

    处理流程：
      1. 繁简体转换（可选）
      2. 全角转半角（可选）
      3. 大小写转换（可选）
      4. 去除填充词（如"呃"、"啊"）（可选）
      5. 去除儿化音（可选）
      6. NSW 规范化（数字、日期、货币等转中文读音）
      7. 非法字符检查（可选）
      8. 智能空格处理（可选）

    使用示例：
      >>> normalizer = TextNorm(to_banjiao=True, remove_erhua=True)
      >>> result = normalizer("2024年1月15日，花费123.45元买了3个苹果")
      "二零二四年一月十五日，花费一百二十三点四五元买了三个苹果"
    """

    def __init__(self,
                 to_banjiao: bool = False,
                 to_upper: bool = False,
                 to_lower: bool = False,
                 remove_fillers: bool = False,
                 remove_erhua: bool = False,
                 check_chars: bool = False,
                 remove_space: bool = False,
                 cc_mode: str = '',
                 ):
        """
        初始化 TextNorm 实例

        参数:
            to_banjiao (bool): 是否将全角字符转换为半角字符
            to_upper (bool): 是否将英文字母转为大写
            to_lower (bool): 是否将英文字母转为小写
            remove_fillers (bool): 是否去除填充词（"呃"、"啊"）
            remove_erhua (bool): 是否去除儿化音
            check_chars (bool): 是否检查非法字符（发现非法字符返回空字符串）
            remove_space (bool): 是否智能处理空格
            cc_mode (str): 繁简体转换模式
                '' - 不转换
                't2s' - 繁体转简体
                's2t' - 简体转繁体
        """
        self.to_banjiao = to_banjiao
        self.to_upper = to_upper
        self.to_lower = to_lower
        self.remove_fillers = remove_fillers
        self.remove_erhua = remove_erhua
        self.check_chars = check_chars
        self.remove_space = remove_space

        # 初始化 OpenCC（繁简体转换引擎）
        self.cc = None
        if cc_mode:
            from opencc import OpenCC  # Open Chinese Convert: pip install opencc
            self.cc = OpenCC(cc_mode)

    def __call__(self, text):
        """
        规范化文本的主入口

        按预定义的顺序应用所有启用的规范化处理。

        参数:
            text (str): 原始文本

        返回:
            str: 规范化后的文本，如果包含非法字符且 check_chars=True 则返回空字符串
        """
        # 1. 繁简体转换
        if self.cc:
            text = self.cc.convert(text)

        # 2. 全角转半角
        if self.to_banjiao:
            text = text.translate(QJ2BJ_TRANSFORM)

        # 3. 转大写
        if self.to_upper:
            text = text.upper()

        # 4. 转小写
        if self.to_lower:
            text = text.lower()

        # 5. 去除填充词（"呃"、"啊"）
        if self.remove_fillers:
            for c in FILLER_CHARS:
                text = text.replace(c, '')

        # 6. 去除儿化音
        if self.remove_erhua:
            text = remove_erhua(text)

        # 7. NSW 规范化（数字、日期、货币等）
        text = normalize_nsw(text)

        # 8. 非法字符检查（可选）
        # 注意：标点转换被注释掉，因为某些 TTS 系统需要保留标点以控制停顿
        # text = text.translate(PUNCS_TRANSFORM)
        if self.check_chars:
            for c in text:
                if not IN_VALID_CHARS.get(c):
                    logger.warning(f'illegal char {c} in: {text}')
                    return ''

        # 9. 智能空格处理
        if self.remove_space:
            text = remove_space(text)

        return text


# ============================================================================ #
# 命令行入口
# 支持多种输入格式：纯文本（txt）、Kaldi 存档格式（ark）、TSV 格式
# ============================================================================ #

if __name__ == '__main__':
    # 创建命令行参数解析器
    p = argparse.ArgumentParser(description='中文文本规范化工具')

    # ---- 规范化选项 ----
    p.add_argument('--to_banjiao', action='store_true',
                   help='将全角字符转换为半角字符')
    p.add_argument('--to_upper', action='store_true',
                   help='将英文字母转换为大写')
    p.add_argument('--to_lower', action='store_true',
                   help='将英文字母转换为小写')
    p.add_argument('--remove_fillers', action='store_true',
                   help='去除填充词，如"呃, 啊"')
    p.add_argument('--remove_erhua', action='store_true',
                   help='去除儿化音，如"他女儿在那边儿 -> 他女儿在那边"')
    p.add_argument('--check_chars', action='store_true',
                   help='检测非法字符，包含非法字符的句子将返回空字符串')
    p.add_argument('--remove_space', action='store_true',
                   help='智能处理空格（保留英文单词间空格）')
    p.add_argument('--cc_mode', choices=['', 't2s', 's2t'],
                   default='', help='繁简体转换模式：t2s（繁体转简体）或 s2t（简体转繁体）')

    # ---- 输入输出选项 ----
    p.add_argument('--log_interval', type=int, default=10000,
                   help='日志输出间隔（处理行数）')
    p.add_argument('--has_key', action='store_true',
                   help="已废弃，请使用 --format ark 代替")
    p.add_argument('--format', type=str,
                   choices=['txt', 'ark', 'tsv'], default='txt',
                   help='输入格式：txt（纯文本）、ark（Kaldi 格式）、tsv（带 TEXT 列的表格）')
    p.add_argument('ifile', help='输入文件路径（UTF-8 编码）')
    p.add_argument('ofile', help='输出文件路径')

    # 解析命令行参数
    args = p.parse_args()

    # 兼容旧的 --has_key 选项
    if args.has_key:
        args.format = 'ark'

    # 创建规范化器实例
    normalizer = TextNorm(
        to_banjiao=args.to_banjiao,
        to_upper=args.to_upper,
        to_lower=args.to_lower,
        remove_fillers=args.remove_fillers,
        remove_erhua=args.remove_erhua,
        check_chars=args.check_chars,
        remove_space=args.remove_space,
        cc_mode=args.cc_mode,
    )

    ndone = 0  # 已处理行数计数器
    # 打开输入文件和输出文件
    with open(args.ifile, 'r', encoding='utf8') as istream, open(args.ofile, 'w+', encoding='utf8') as ostream:
        if args.format == 'tsv':
            # ---- TSV 格式处理 ----
            # TSV 是制表符分隔的表格格式，需要处理包含 "TEXT" 列的文件
            reader = csv.DictReader(istream, delimiter='\t')
            if 'TEXT' not in reader.fieldnames:
                raise ValueError(f"输入 TSV 文件缺少 'TEXT' 列，实际列名: {reader.fieldnames}")
            # 输出表头
            print('\t'.join(reader.fieldnames), file=ostream)

            # 逐行处理
            for item in reader:
                text = item['TEXT']

                if text:
                    text = normalizer(text)

                if text:
                    # 规范化成功，更新该行的 TEXT 列并输出
                    item['TEXT'] = text
                    print('\t'.join([item[f]
                          for f in reader.fieldnames]), file=ostream)

                ndone += 1
                # 按日志间隔输出进度
                if ndone % args.log_interval == 0:
                    print(f'text norm: {ndone} lines done.',
                          file=sys.stderr, flush=True)
        else:
            # ---- TXT/ARK 格式处理 ----
            for l in istream:
                key, text = '', ''
                if args.format == 'ark':
                    # KALDI 存档格式：每行格式为 "key text"
                    cols = l.strip().split(maxsplit=1)
                    key, text = cols[0], cols[1] if len(cols) == 2 else ''
                else:
                    # 纯文本格式：每行就是文本内容
                    text = l.strip()

                if text:
                    text = normalizer(text)

                if text:
                    # 输出结果
                    if args.format == 'ark':
                        print(key + '\t' + text, file=ostream)
                    else:
                        print(text, file=ostream)

                ndone += 1
                if ndone % args.log_interval == 0:
                    print(f'text norm: {ndone} lines done.',
                          file=sys.stderr, flush=True)

    # 输出最终统计
    print(f'text norm: {ndone} lines done in total.',
          file=sys.stderr, flush=True)
