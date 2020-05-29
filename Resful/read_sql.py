# import pandas
# import matplotlib as plt
# # import sklearn
# import sys
# import random
#import pyodbc

# pymssql.connect(host="192.168.1.108:1433",user="sa",password="powerSIS#123",database="alert",charset="utf8")

# conn = pyodbc.connect(r'DRIVER={SQL Server Native Client 11.0};SERVER=test;DATABASE=test;UID=user;PWD=password')
# conn = pyodbc.connect(r'DRIVER={{SQL Server Native Client 10.0};SERVER=192.168.1.108;DATABASE=alert;UID=sa;PWD=powerSIS#123')
import pymssql
import json
class MSSQL:
    def __init__(self,host,user,pwd,database):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = database

    def __GetConnect(self):
        """
        得到连接信息
        返回: conn.cursor()
        """
        if not self.db:
            raise(NameError,"没有设置数据库信息")
        self.conn = pymssql.connect(host=self.host,user=self.user,password=self.pwd,database=self.db,charset="utf8")
        cur = self.conn.cursor()
        if not cur:
            raise(NameError,"连接数据库失败")
        else:
            return cur

    def ExecQuery(self,sql):
        """
        执行查询语句
        返回的是一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段
        """
        cur = self.__GetConnect()
        cur.execute(sql)
        resList = cur.fetchall()

        #查询完毕后必须关闭连接
        self.conn.close()
        return resList

    def ExecNonQuery(self,sql):
        """
        执行非查询语句

        调用示例：
            cur = self.__GetConnect()
            cur.execute(sql)
            self.conn.commit()
            self.conn.close()
        """
        cur = self.__GetConnect()
        cur.execute(sql)
        self.conn.commit()
        self.conn.close()

def get_model_by_ID(model_id):
    ms = MSSQL(host="192.168.0.146:1433", user="sa", pwd="powerSIS#123", database="alert")
    resList = ms.ExecQuery("SELECT * FROM [alert].[dbo].[Model_CFG] where \"model_id\"="+str(model_id))
    model_info=json.loads(resList.tolist())
    return json.loads(resList[0][6])["para"]

model=get_model_by_ID(5)
print(model)