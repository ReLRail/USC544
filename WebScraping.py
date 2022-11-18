import re
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen


def get_soup(url):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'
    }
    return BeautifulSoup(urlopen(Request(url, headers=header)), 'html.parser')


def find_Name(soup):
    name_list = []
    content = soup.find_all(class_="link-blanc")
    # names = content.find()
    for names in content:
        name = names.text
        name_list.append(name)

    return name_list


def find_label(soup):
    contents = soup.find_all("script")
    label_list = []
    data_list = []
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
            label_list = re.findall(r'\d+', label_string)
            data_string = script_list[data_index]
            data_list = re.findall(r'\d+', data_string)
            break
    return [label_list, data_list]


def get_action_list(action_content):
    action_list = []
    for act in action_content:
        get_alt = act.find('img', alt=True)
        action_list.append([get_alt["alt"], act.text])
    return action_list


def find_action(soup):
    blue_contents = soup.find_all(class_="blue_action")
    red_contents = soup.find_all(class_="red_action")
    action_dict = {}
    get_blue_list = get_action_list(blue_contents)
    action_dict['blue'] = get_blue_list
    get_red_list = get_action_list(red_contents)
    action_dict['red'] = get_red_list

    return action_dict


def display(soup):
    name_list = find_Name(soup)
    label_list = find_label(soup)
    actions_dict = find_action(soup)
    print("the player name")
    print(name_list)
    print()
    print("the lable and data for the chart")
    print(label_list)
    print()
    print("the action for two teams")
    print(actions_dict)


# change in url in here
url = "https://gol.gg/game/stats/44417/page-game/"
# page = requests.get(URL)
# soup = BeautifulSoup(page.content, "html.parser")
soup = get_soup(url)
# name_list = find_Name(soup)
# label = find_label(soup)
# actions = find_action(soup)
display(soup)