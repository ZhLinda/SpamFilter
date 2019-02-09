#!/usr/bin/env python
# -*- coding:utf-8 -*-
from curses import ascii
'''
处理汉字的工具：判断unicode是否是汉字、数字、英文字母，或者其他字符;以及全半角转换
'''
def is_Chinese(ch):
    """return True if ch is Chinese character.
        full-width puncts/latins are not counted in.
        """
    x = ord(ch)
    # CJK Radicals Supplement and Kangxi radicals
    if 0x2e80 <= x <= 0x2fef:
        return True
    # CJK Unified Ideographs Extension A
    elif 0x3400 <= x <= 0x4dbf:
        return True
    # CJK Unified Ideographs
    elif 0x4e00 <= x <= 0x9fbb:
        return True
    # CJK Compatibility Ideographs
    elif 0xf900 <= x <= 0xfad9:
        return True
    # CJK Unified Ideographs Extension B
    elif 0x20000 <= x <= 0x2a6df:
        return True
    else:
        return False

def is_number(ch):  #judge whether the character is a number or not
    if ch >= u'\u0030' and ch <= u'\u0039':
        return True
    else:
        return False

chinese_number = (u'十', u'百', u'千', u'万', u'亿', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'零', u'几')

def is_Chinese_number(ch):
    return ch in chinese_number

def is_alpha(ch):   #judge whether the character is English or not
    if (ch >= u'\u0041' and ch <= u'\u005a') or (ch >= u'\u0061' and ch <= u'\u007a'):
        return True
    else:
        return False

def is_punctuation(ch):
    x = ord(ch)
    # in no-formal literals, space is used as punctuation sometimes.
    if x < 127 and ascii.ispunct(x):
        return True
    # General Punctuation
    elif 0x2000 <= x <= 0x206f:
        return True
    # CJK Symbols and Punctuation
    elif 0x3000 <= x <= 0x303f:
        return True
    # Halfwidth and Fullwidth Forms
    elif 0xff00 <= x <= 0xffef:
        return True
    # CJK Compatibility Forms
    elif 0xfe30 <= x <= 0xfe4f:
        return True
    else:
        return False

def is_other(ch): #判断是否非汉字、数字、英文字母
    #if not (is_Chinese(ch) or is_number(ch) or is_alpha(ch) or is_punctuation(ch)):
    if not (is_Chinese(ch) or is_number(ch) or is_alpha(ch)):
        return True
    else:
        return False

def B2Q(ch): #半角转全角
    inside_code = ord(ch)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return ch
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return unichr(inside_code)

def Q2B(ch): #全角转半角
    inside_code = ord(ch)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return ch
    return unichr(inside_code)

def del_nums_chars_puncts(ustring):
    retlist = []
    tmp = []
    for ch in ustring:
        if is_number(ch) or is_alpha(ch) or is_punctuation(ch):
            if len(tmp) == 0:
                continue
            else:
                retlist.append("".join(tmp))
                tmp = []
        else:
            tmp.append(ch)
    if len(tmp) != 0:
        retlist.append("".join(tmp))
    msg = ''
    for key in retlist:
        msg += key

    return msg.encode('utf-8')

def string_to_list(ustring): #去除所有被is_other()判为真的字符

    retlist = []
    tmp = []
    for ch in ustring:
        if is_other(ch):
            if len(tmp) == 0:
                continue
            else:
                retlist.append("".join(tmp))
                tmp = []
        else:
            tmp.append(ch)
    if len(tmp) != 0:
        retlist.append("".join(tmp))
    #cnt = len(ustring) - len(retlist)

    return retlist#, cnt