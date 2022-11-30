import re
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import pandas as pd
from datetime import datetime
def get_soup(url):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'
    }
    return BeautifulSoup(urlopen(Request(url, headers=header)), 'html.parser')


def find_Name(soup):
    name_list = []
    content = soup.find_all(class_="champion_icon rounded-circle")
    # names = content.find()
    for names in content:
        name = names["alt"]
        name_list.append(name)

    return name_list


def find_label(soup):
    contents = soup.find_all("script")
    data_list = []
    label=0
    for content in contents:
        if "var golddatas" in content.text:
            script = content.text
            script = script.split('var golddatas = ')[1]
            script = script.replace("/t", "")
            script = script.replace("/n", "")
            script_list = script.strip().split(": ")
            label_index = 0
            data_index = 0
            for index, data in enumerate(script_list):
                if "labels" in data:
                    label_index = index + 1
                if "data" in data:
                    data_index = index + 1
            label_string = script_list[label_index]
            label_list =  list(map(int,re.findall(r'\d+', label_string)))
            data_string = script_list[data_index]
            data_list = list(map(int, re.findall(r'-?\d+\.?\d*', data_string)))
            break
    # if label_list[-1]==label_list[-2]:
    #     label=label_list[-1]+1
    # else:
    #     label=label_list[-1]
    # data_list.insert(0,label)
    return data_list


def get_action_list(action_content,types):
    action_list = []
    for act in action_content:
        get_alt = act.find('img', alt=True)
        action_list.append([datetime.strptime(act.text,'%M:%S'), types, get_alt["alt"].lower()])
    return action_list


def find_action(soup):
    blue_contents = soup.find_all(class_="blue_action")
    red_contents = soup.find_all(class_="red_action")
    get_blue_list = get_action_list(blue_contents,'b')
    
    get_red_list = get_action_list(red_contents,'r')
    actions = sorted(get_blue_list+get_red_list)
    return actions


def get_data(data):
    name_list = []
    label_list =[]
    actions_dict =[]
    for i,j in enumerate(data['data_link']):
        
        soup = get_soup(j)
        name_list.append(find_Name(soup))
        label_list.append(find_label(soup))
        actions_dict.append(find_action(soup))
    data['champions']=name_list
    data['gold']=label_list
    data['action']=actions_dict
    # print("the player name")

    # print()
    # print("the lable and data for the chart")

    # print()
    # print("the action for two teams")

    return data
