#-*- coding: UTF-8 -*-
#Auther HJ
#把Html文件转成Txt格式文件

from HTMLParser import HTMLParser
from re import sub
from sys import stderr
from traceback import print_exc
import  markupbase


class _DeHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.__text = []

    def handle_data(self, data):
        text = data.strip()
        if len(text) > 0:
            text = sub('[ \t\r\n]+', ' ', text)
            self.__text.append(text + ' ')

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self.__text.append('\n\n')
        elif tag == 'br':
            self.__text.append('\n')

    def handle_startendtag(self, tag, attrs):
        if tag == 'br':
            self.__text.append('\n\n')

    def text(self):
        return ''.join(self.__text).strip()


def dehtml(text):
    try:
        parser = _DeHTMLParser()
        parser.feed(text)
        parser.close()
        return parser.text()
    except:
        print_exc(file=stderr)
        return text


def main():
  #  text = r'''''
  #  url="E:\罗一夫课题\DeepLearning ER\\00213237_1\\00213237_34763_20170126092657_1.html"
   # htmlfile = open(url, 'r')
  #  htmlpage = htmlfile.read()
    htmlTotxt('E:/罗一夫课题/DeepLearning ER/00213237_1/','00213237_34763_20170126092657_1')
  #  print(dehtml(htmlpage))
def htmlTotxt(path,url):
    e = open(path+url + ".txt", 'w')
    url=path+url+'.html'
    htmlfile = open(url,'r')
    htmlpage = htmlfile.read()
    str = dehtml(htmlpage)
    e.write(str)
    e.close()

if __name__ == '__main__':

    main()