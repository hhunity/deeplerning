from pymongo import MongoClient
import datetime
import pandas as pd

class mongoDBW():
    def __init__(self,dbname=''):
        self.dbname    = dbname
        self.beforeBit = 0
        self.beforeAsk = 0
        
        if self.isRecordDb():
            self.client = MongoClient('mongodb://localhost:27017/')
            self.db     = self.client[self.dbname]
    
    def close(self):
        if self.isRecordDb():
            self.db.client.close()
        
    def isRecordDb(self):
        if len(self.dbname) > 0:
            return True
        else:
            return False
        
    def insert_one(self,collection,post):

        if self.isRecordDb():
            self.db[collection].insert_one(post)

    def find(self,collection,start:datetime,end:datetime):

        if self.isRecordDb():
            # クエリでデータ取得
            query = {
                "time": {
                    "$gte": start,
                    "$lte": end
                }
            }
            co = self.db[collection].find(filter=query)
            df = pd.DataFrame(list(co))
            #id削除
            df = df.drop(columns='_id')
            # UTCで格納されているので、JSTに変更して時差情報削除して返却
            df['time'] = pd.to_datetime(df['time'],utc=True)
            df['time'] = df['time'].dt.tz_convert('Asia/Tokyo')
            df['time'] = df['time'].dt.tz_localize(None)
            
            return df
        
        return pd.DataFrame()

    def insertTick2(self,time:datetime,bid,ask,sp=0,bcheckSame=True):
        
        if bcheckSame:
            if (self.beforeBit == bid) or (self.beforeAsk == ask):#and?
                #bid,askが同じなら格納しない
                return

        self.beforeBit = bid
        self.beforeAsk = ask

        post = {
            'time'     : time,
            'bid'      : bid,
            'ask'      : ask,
            'sp'       : sp,
        }

        self.insert_one('tick2',post)

    def insertTick(self,time:datetime,bid,ask,bidsize=0,asksize=0,bcheckSame=True):
        
        if bcheckSame:
            if (self.beforeBit == bid) or (self.beforeAsk == ask):#and?
                #bid,askが同じなら格納しない
                return

        self.beforeBit = bid
        self.beforeAsk = ask

        post = {
            'time'     : time,
            'bid'      : bid,
            'ask'      : ask,
            'bids'     : bidsize,
            'asks'     : asksize,
        }

        self.insert_one('tick',post)

    def insertTransaction2(self,time:datetime,order_action,trade_action
                           ,order_price=0,get_pips=0,daily_pnl=0,total_asset=0):

        post = {
            'time'          : time,
            'order_action'  : order_action,
            'trade_action'  : trade_action,
            'order_price'   : order_price,
            'get_pips'      : get_pips,
            'daily_pnl'     : daily_pnl,
            'total_asset'   : total_asset,
        }

        self.insert_one('transaction2',post)

    def insertTransaction(self,time:datetime,action1,action2,price=0,info1=0,info2=0,info3=0,info4=0):

        post = {
            'time'     : time,
            'action1'  : action1,
            'action2'  : action2,
            'price'    : price,
            'info1'    : info1,
            'info2'    : info2,
            'info3'    : info3,
            'info4'    : info4,
        }

        self.insert_one('transaction',post)

    def insertReport(self,time:datetime,balance,unrPnl,Pnl,num,cost):
        
        post = {
            'time'      : time,
            'balance'   : balance,
            'unrPnl'    : unrPnl,
            'Pnl'       : Pnl,
            'num'       : num,
            'cost'      : cost,
        }

        self.insert_one('report',post)

    def findTick(self,startTime:datetime,days,record='tick'):
        
        sTime = startTime - datetime.timedelta(days=days)
        eTime = startTime 
        
        #sTime以上,eTime未満
        return self.find(record,query={"time":{"$gte":sTime,"$lt":eTime}})
    
    def findtransaction(self,startTime:datetime,days,record='transaction'):
        
        sTime = startTime - datetime.timedelta(days=days)
        eTime = startTime
        print(sTime,eTime)
        #sTime以上,eTime未満
        return self.find(record,query={"time":{"$gte":sTime,"$lt":eTime}})

    def findreport(self,startTime:datetime,days):
        
        sTime = startTime - datetime.timedelta(days=days)
        eTime = startTime
        print(sTime,eTime)
        #sTime以上,eTime未満
        return self.find('report',query={"time":{"$gte":sTime,"$lt":eTime}})

if __name__ == "__main__":
    db = mongoDBW('USDJPY3')

    import commonLib as cl
    import datetime
    import dataframeLib as dl

    jtime   = cl.japanTime()
    nowtime = jtime.getJapanTime()
    settime = nowtime+datetime.timedelta(days=1)
    df = db.find("tick2",datetime.datetime(2025,1,10,0,0,0),datetime.datetime(2025,1,10,23,59,59))
    df = dl.make_ohlc_from_ticks(df)
    #df = db.findtransaction(nowtime,1)
    print(df)
