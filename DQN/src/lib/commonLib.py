import os
import datetime
import investpy
import pandas as pd
import csv
import time
from dateutil.tz import gettz
from datetime import timedelta

class eventClas():
    def __init__(self,event,object1,object2=0,object3=0,object4=0):
        self.event    = event
        self.object   = object1
        self.object2  = object2
        self.object3  = object3
        self.object4  = object4

def sleepConsol(sleepTime,str=''):
    print('\033[2K\033[G',end='')
    for i in range(1, sleepTime + 1):
        pro_bar = ('=' * i) + (' ' * (sleepTime - i))
        print(f'\r [{pro_bar}] {str} {i}/{sleepTime} sec', end='')
        time.sleep(1)
    print('\033[2K\033[G',end='')

def changeFormat(time:datetime):
    return time.strftime('%Y/%m/%d %H:%M:%S')

def changeFormatForTick(time:datetime):
    return time.strftime('%Y/%m/%d %H:%M:%S.%f')

def changeFormatForFile(time:datetime):
    return time.strftime('%Y%m%d_%H%M%S')

def changeFormatDayOnly(time:datetime):
    return time.strftime('%Y/%m/%d')

def changeFormatDayHMOnly(time:datetime):
    return time.strftime('%Y/%m/%d %H:%M')

def getUtctimeWithtimeJST(time:datetime):
    return datetime.datetime.timestamp(time)

def changeTimeZoneToJST(time:datetime):
    return time.astimezone(gettz('JST'))

class price:
     def __init__(self,bidprice=0,askprice=0,bidsize=0,asksize=0) -> None:
         self.bitprice = bidprice
         self.askprice = askprice
         self.bidsize = bidsize
         self.asksize = asksize
         self.utime   = 0
         self.sp      = 0
         self.jtime   = ''
         
class japanTime:
    def __init__(self):
        t_delta = datetime.timedelta(hours=9)
        self.jst = datetime.timezone(t_delta, 'JST')

    def getJapanTime(self):
        return datetime.datetime.now(self.jst)

    def convUtcJptime(self,timestamp):
        utc = datetime.datetime.fromtimestamp(timestamp/1000,self.jst)
        milliseconds = timestamp % 1000
        return utc+timedelta(milliseconds=milliseconds)  #utc.strftime('%Y-%m-%d %H:%M:%S') + f'.{milliseconds:03d}'

class tradeTime:
    TIME_NON = 0
    TIME_ECONOMIC = 1
    TIME_RESET = 2

    def __init__(self,kind,time:datetime,option) -> None:
        self.kind   = self.TIME_NON
        self.time   = time
        self.option =option

    def __str__(self) -> str:
        return f'kind:{self.kind} {self.time.month}/{self.time.day} {self.time.hour:02d}:{self.time.minute:02d} op:{self.option}'

def getday(time:datetime):
    return time.strftime('%-m/%-d')

class economicTime:
    def __init__(self):
        self.timelist = []
        self.csvpath = ''

    def setCsv(self,csvpath):
        self.csvpath = csvpath
    
    def appendEconomicTime(self,kind,option,day:datetime):
        t = tradeTime(kind,day,option)
        self.timelist.append(t)       

    def getEconomicTime(self,readcsvpath=None):
        
        if readcsvpath == None:
            try:
                url = 'https://www.gaikaex.com/gaikaex/mark/calendar/' #みんかぶFXの経済指標URLを取得
                dfs = pd.read_html(url) #テーブルのオブジェクトを生成
                dfs = dfs[0]
            except:
                print('get ecnomict time error -> skip')
                return
        else:
            dfs  = pd.read_csv(readcsvpath,index_col=0)
        

        if len(self.csvpath) > 0:
            dfs.to_csv(self.csvpath)
        
        dfs = dfs[dfs['重要度'].isin(['★★★','★★'])]
        dfs = dfs[dfs['国'].isin(['米国','日本','ユーロ'])]
        dfs = dfs.drop_duplicates(subset=['発表日','時刻'])
        #print(dfs)

        for i,row in dfs.iterrows():
            daystr = row['発表日']
            daystr = daystr[:daystr.find(' (')]
            day    = datetime.datetime.strptime(daystr,'%m/%d')
            
            timestr = row['時刻']
            pos     = timestr.find(':')

            if '--' in timestr:
                continue
            
            delta = datetime.timedelta(hours=int(timestr[:pos]),minutes=int(timestr[pos+1:]))
            day = day+delta

            t = tradeTime(tradeTime.TIME_ECONOMIC,day,10)
            self.timelist.append(t)

    def clear(self):
        self.timelist.clear()

class tradeTimeClass:

    def __init__(self,sleep_s=2,sleep_e=7):
        self.clear()
        self.sleep_start = sleep_s
        self.sleep_end   = sleep_e

    def clear(self):
        self.latestLinehor = 999
    
    def isWeekendLogOut(self,time:datetime):
        result = False

        w = int(time.weekday())
        h = int(time.hour)
        
        # 5:土曜 次以降 6:日曜　終日　7:月曜 ７時以降
        if   (w == 6) or \
            ((w == 5) and (h >= self.sleep_start)) or\
            ((w == 0) and (h <= self.sleep_end)):
            result = True
        
        return result

    def isSleepTime(self,time:datetime):
        result = False

        h = int(time.hour)

        if (h >= self.sleep_start and h <= self.sleep_end):
            result = True

        return result

class recordeClass:
    def __init__(self,bcsv):
        #CSVファイルの準備
        self.referencejTime = japanTime().getJapanTime()
        self.jtimestr       = changeFormatForFile( self.referencejTime )

        if bcsv:
            self.bcsv=True
            pricecsvPath=os.path.dirname(__file__)+'/../csv/'+ self.jtimestr +'Stop.csv'
            tradecsvPath=os.path.dirname(__file__)+'/../csv/'+ self.jtimestr +'StopTra.csv'

            self.pricecsv_file =  open( pricecsvPath,'w')
            self.tradecsv_file =  open( tradecsvPath,'w')
            self.pricecsvw = csv.writer(self.pricecsv_file)
            self.tradecsvw = csv.writer(self.tradecsv_file)
        else:
            self.bcsv=False
            self.pricecsv_file = None
            self.tradecsv_file = None
            self.pricecsvw = None
            self.tradecsvw = None
    def close(self):
        if self.pricecsv_file is not None:
            self.pricecsv_file.close()
        if self.tradecsv_file is not None:
            self.tradecsv_file.close()

if 0:
    e = economicTime()
    e.getEconomicTime()
    for e in e.timelist:
        print(e)
    jt = japanTime()
    print(jt.convUtcJptime(98888))
    print(jt.getJapanTime())