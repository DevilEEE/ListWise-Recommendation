# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:25:00 2019

@author: v_fdwang
"""
import os
import json

class DataPre(object):
    def __init__(self, raw_file_path='F:/data/test_6_20181220', dic_path = 'F:/data/dic.txt', reset_dic=False):
        self.raw_file_path = raw_file_path
        self.data_path = raw_file_path + '/data.txt'
        self.reset_dic = reset_dic
        self.dic_path = dic_path
        self.cate2value = {}
        self.init_dic()
        self.message = None
        
        if not os.path.isfile(self.data_path):
            self.send()
            self.init_dic()
            self.data_process()
        self.show_message()
    
    def init_dic(self):
        if not self.reset_dic and os.path.isfile(self.dic_path):
            with open(self.dic_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.cate2value = json.loads(line.strip())
                    if not isinstance(self.cate2value, dict):
                        raise ValueError("Except a dict!")
        
    
    def data_process(self):
        part_list = [self.raw_file_path + '/' + w for w in os.listdir(self.raw_file_path) if 'part' in w]
        value2cate = {}
        value = 0

        total_data = []
        for part in part_list:
            train_data = []
            with open(part, 'r', encoding='utf-8') as f:
                for line in f:
                    content = json.loads(line.strip())
                    #print(content)
                    sessionid = content['sessionid']
                    sessionObject = content['sessionObject']
                    push_list = sessionObject.split('|pushseq|')
                    for push in push_list:
                        pushObject = json.loads(push)['pushObject']
                        item_list = pushObject.split('|itemseq|')
                        train_x = []
                        train_y = []
                        feedid_list = []   # no duplicate feedid
                        for item in item_list:
                            try:
                                item = json.loads(item)
                                itemObject = json.loads(item['itemObject'])
                            except:
                                continue
                            
                            try:
                                feedid = itemObject['feedid']
                                if feedid in feedid_list:
                                    continue
                                feedid_list.append(feedid)
                            except:
                                continue
                            # get for train data
                            try:
                                category = itemObject['category']
                                res = float(itemObject['video_play_time'])/float(itemObject['video_total_time'])
                                res = min(res, 1.0)
                            except:
                                continue
                            if category not in self.cate2value:
                                self.message = 'Better to reset your dic, new cate come!'
                                self.cate2value[category] = value
                                value2cate[value] = category
                                value += 1
                            train_x.append(self.cate2value[category])
                            train_y.append(res)
                        try:
                            train_data.append((train_x, sum(train_y)/len(train_y)))
                        except:
                            continue
            total_data.extend(train_data)
            
        if self.reset_dic or not os.path.isfile(self.dic_path):
            with open(self.dic_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(self.cate2value))
        
        with open(self.data_path, 'w', encoding='utf-8') as f:
            for line in total_data:
                temp = [str(w) for w in line[0]]
                f.write(json.dumps({'feature':','.join(temp), 'label':str(line[1])}) + '\n')
                
    def get_data_path(self):
        return self.data_path
    
    def get_dic_path(self):
        return self.dic_path
    
    def show_message(self):
        if self.message is None:
            print('No new cate come, no need to renew your cate dic!')
        elif not self.reset_dic:
            print(self.message)
        else:
            print("Dict reset over!")
        print("Data Preprocessed Done!")
    
    def send(self):
        print("Data Preprocessing!")
        





    
