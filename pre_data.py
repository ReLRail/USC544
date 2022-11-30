import datetime
import pandas as pd
import re
import pickle
from WebScraping import get_data
def check_name(names):
    wrong={'zeri':['zerry'],'kennen':['cannon','kenneth'],'nocturne':['noctur'],'azir':['azeal'],'miss fortune':['mf'],'soraka':['sorako'],
    'jiejie':['jj','jojo','ginger'],'beryl':['barrel'],'deft':['death','defton','debt',"def's",'depth','devt'],'flandre':['andre','flandra','flange','fondra','Padre'],'100':['hundredth','hundred'],'closer':['moser'],'t1':['tijuan','skt'],'gumayusi':['guinea you see','gumiuci','gomeshin','gumayushi','gummy','kuma','gumiyushi','gumushi','kumushi','kumiyushi','kumi','Goomba'],
    'keria':['kyrie','kerry','karius','kerria','harry'],'oner':['owner','hornhorn'],'odoamne':['Odo Omni','odumne','odorama','otto','odwame','one day','odo','woman'],'malrang':['mauro','mal','mal rang','mal ryan','mal ring','mal rank','mallory','maurang','maurango','mauron'],'kanavi':['cannabi','could not be','cannavi','canaveri','kanabi','kenobi','navy','knavy','konami'],
    'meiko':['mako'], 'tl':['team liquid','liquid'], 'c9':['cloud'],'ssumday':['someday','somebody'],'blaber':['blabber','blapper','flapper','flabber'],'zven':['sven','zen','sivir','silver'],'abbedagge':['abu','abaddaga'],'rge':['rouge','roe','bro'],'facker':['baker','picker'],
    'larssen':['lars','larson','so lost'],'yagao':['ikao','yagal','the girl'],'comp':['khan','cops'],'trymbi':['trimby'],'369':['3-6'],'gen':['genji','genjir'],'chovy':['jovi','chovey','chobi','shovey','trophy','toby','show me'],'lehends':['landon','lahenz'],'zeus':['zayus','zayo','diaz'],
    'dk':['domood','dom one','domon kia','domonkia','domo','dom juan kia','domin','domo kia','domon','dom on','damon'],'nuguri':['nugury','nogory','nuggery','naugery'],'showmaker':['shoemaker'],'kellin':['callan','kaden','killen','kellen'],
    'deokdam':['dot tom','doctor'],'jojopyun':['jojobian','jojo pyon','jojo pian','jojo'],'santorin':['santhorn','santorum','santorini'],'bwipo':['bobo','buppo','blippo','wifo','buffalo','wipo'],'vulcan':['vulcan'],'kaori':['curry','calorie','corey'],
    'hans sama':['han sama','han salma','hansa','hansama','han summer','hans summer','han solo','han sonic','han samuel','han','hans'],'brokenblade':['broken plate','broken blade','broken blades'],'jankos':['yankos','yankov','yankovsk','gankos','yanko'],
    'caps':['cap series',"cap's"],'flakkad':['flacco','flak','flack','flakker','flaca','flaccid'],'targamas':['targets','targamous','targamus','target mass'],'kingen':['kj','king','kingdom','kingman'],'zeka':['zika','zach','zecco','zaka','zekka','zeko','zecca','zecker','zeca','zeke'],'drx':['d-rex','trx'],
    'pyosik':['kyoshik','pieology','yoshik','piyoshik','pyoshik','fioshik','shuriken','pioshik'],'scout':['scott','scouted','scouts'],'viper':['piper'],'jackylove':['jackie'],'tian':['tien'],'tes':['tom','tso','top esports','ts','topside','top b sports','top eastwood','toppy sports','toppies','top'],
    'knight':['night','tonight'],'fnc':['fanatic','fnatic','phonetic','fnatica'],'wunder':['wonder','wander'],'razork':['razzler','razorc','razzorax','razak','razzock','razark','rasok','razzler','razorque','razor','razzle'],'humanoid':['human only','human it','human would','humor','human'],'hylissang':['hilasank','hilisang','hilly','hilusang','hillside'],
    'upset':['offset'],'inspired':['inspire'],'g2':['gt'],'sgb':['saigon buffalo','taigon buffalo','buffalo','saigon'],'hasmed':['hazmi','hassman','hazmat','hazmed','husband'],'armut':['armored','armor','ahmed'],'beanj':['bj','dj','pj'],'elyoya':['ayoya','a yoya','yoya'],'froggy':['frankie'],'nisqy':['misky','nisky','niski'],'taki':['tacky'],
    'unforgiven':['unforgiving','forgiven','forgiving'],'corejj':['for jj','core jj','jj'],'johnsun':['johnson'],'fly':['flyquest'],'josedeodo':['jose'],'spica':['spika'],'philip':['filipino','phillip'],'toucouille':['takuya','takoy'],'aphromoo':['afrimo'],'breathe':['breath','bree'],'xiaohu':['she who','yahoo'],
    'isg':['isaurus','isrus','issaurus','estruss'],'add':['80d'],'seiya':['sayer','saya'],'grell':['girl','growl','grill'],'gavotto':['gravity','gavoto','caboto'],'yaharong':['yaharan','yaharang'],'evi':['evie'],'yutapon':["you'd upon", "feud upon", 'utopone','unipone','upon','eudapon','you depone'],
    'byg':['beyond'],'husha':['hoshi','kushi','hush','pusha','who she'],'likai':['lee kai','lee kay','lakai'],'wako':['wacko','wackos'],'minji':['ninji','minjito'],'brance':['brantstone','branson','brands','branch'],'tinowns':['tenones','tenos','tinones','10 owns','10 Owens'],
    'ceos':['theos','seos'],'croc':['kroc'],'lll':['loud'],'shunn':['shun','sean','shown','shen']
    }
    for i in wrong:
        if i in names:
            names+=wrong[i]
    return names

def replace_name(s,r_name,b_name):
    for i in r_name:
        s=s.replace(i,'teamred')
    for i in b_name:
        s=s.replace(i,'teamblue') 
    return s
def preparedata(data):
    if data['series'] in ['WC2022','LCS2022','WCP2022']:
        cham_table=pd.read_excel(data['series']+'.xlsx')
        init_rate1=0
        init_rate2=0
        for j,i in enumerate(data['champions']):
            if j <5:
                a=cham_table.loc[cham_table['champion']== i]
                init_rate1+=float(cham_table.loc[cham_table['champion'] == i]['rate'])
            else:
                init_rate2+=float(cham_table.loc[cham_table['champion'] == i]['rate'])
        rate_sum=(init_rate1+init_rate2)
        init_rate1=init_rate1/rate_sum
        init_rate2=init_rate2/rate_sum
    else:
        init_rate1=0.5
        init_rate2=0.5
    b_names=check_name([data["team_one_player4"].lower(), data["team_one_player2"].lower(), data["team_one_player5"].lower(), data["team_one_player1"].lower(), data["team_one_color"].lower(), data["team_one_name"].lower(), data["team_one_player3"].lower()])
    r_names=check_name([data["team_two_player9"].lower(), data["team_two_player6"].lower(), data["team_two_color"].lower(), data["team_two_name"].lower(), data["team_two_player7"].lower(), data["team_two_player8"].lower(), data["team_two_player10"].lower()])
    names=b_names+r_names
    if pd.notna(data['text2']):
        data['text']+=data['text2']
    


    data['text']=data['text'].replace('\n',' ').replace('\r',' ')
    video_start,game_start=data['start_time'].split(',')
    video_start=datetime.datetime.strptime(video_start, '%M:%S')
    game_start=datetime.datetime.strptime(game_start, '%M:%S')
    time_break=re.findall(r'\d{1,2}\:\d{1,2}', data['text'])
    sentences=re.split(r'\d{1,2}\:\d{1,2}', data['text'])


    
    #ignore sentence before game start
    useful_sent=[]
    useful_label=[]
    test_label=[]
    devider=8000
    p_time=datetime.datetime.strptime('0:00', '%M:%S')
    max_time=len(data['gold'])-1
    pre_rate1=init_rate1
    pre_rate2=init_rate2
    rate1=init_rate1
    rate2=init_rate2
    p_minute=0
    t_s=''
    for i in range(len(time_break)):
        c_time=datetime.datetime.strptime(time_break[i], '%M:%S')
        gaps=c_time-p_time
        if c_time>=video_start:

            # s=sentences[i+1]
            
            # for j in names:
            #     if j == 'gen':
            #         j='gen '
            #     j=' '+j

            #     if j in s:
            #         if j == ' top':
            #             if ('top lane' not in s) and ('top side' not in s):
            #                 # s=replace_name(s,r_names,b_names)
            #                 t_s=t_s+s
            #                 break
            #         elif j==' wonder':
            #             if ('i wonder' not in s):
            #                 # s=replace_name(s,r_names,b_names)
            #                 t_s=t_s+s
            #                 break  
            #         else:
            #             # s=replace_name(s,r_names,b_names)
            #             t_s=t_s+s
            #             break
            
            c_game_time=c_time-video_start+game_start
            c_minutes=c_game_time.minute
            if c_minutes+1>max_time:
                break
            if p_minute!=c_minutes and c_minutes!=0:

                # avg_gold=((data['gold'][c_minutes+1]-data['gold'][c_minutes]))/(devider)
                # if avg_gold<0:
                #     rate2=rate2-avg_gold
                # else:
                #     rate1=rate1+avg_gold
                # rate_sum=(rate1+rate2)
                # rate1=rate1/rate_sum
                # rate2=rate2/rate_sum
                # useful_sent.append(t_s)
                # useful_label.append([rate1-pre_rate1])
                # test_label.append([rate1])
                pre_rate1=rate1
                pre_rate2=rate2
                # t_s=''

            
            if data['action'] and c_game_time>=data['action'][0][0]:
                action_rate=0
                action=data['action'][0][2]
                if 'blood' in action:
                    action_rate=0.1
                elif 'elder' in action:
                    action_rate=0.5
                elif 'tower' in action:
                    action_rate=0.1
                elif 'drake' in action:
                    action_rate=0.1
                elif 'rift' in action:
                    action_rate=0.1
                elif 'nashor' in action:
                    action_rate=0.5
                if data['action'][0][1]=='b':
                    rate1+=action_rate
                else:
                    rate2+=action_rate
                rate_sum=(rate1+rate2)
                rate2=rate2/rate_sum
                rate1=rate1/rate_sum
                data['action'].pop(0)
            devider=8000
            if gaps.seconds==0 :
                avg_gold=0
            else:
                #avg
                # avg_gold=((data['gold'][c_minutes+1]+data['gold'][c_minutes]))/(2*devider)
                #partial_change
                avg_gold=((data['gold'][c_minutes+1]-data['gold'][c_minutes])*gaps.seconds/60)/(devider)
                #avg_change
                # avg_gold=((data['gold'][c_minutes+1]-data['gold'][c_minutes]))/(devider)
            if avg_gold<0:
                rate2=rate2-avg_gold
            else:
                rate1=rate1+avg_gold


            rate_sum=(rate1+rate2)
            rate1=rate1/rate_sum
            rate2=rate2/rate_sum

            s=sentences[i+1]
            for j in names:
                if j == 'gen':
                    j='gen '
                j=' '+j

                if j in s:
                    if j == ' top':
                        if ('top lane' not in s) and ('top side' not in s):
                            # s=replace_name(s,r_names,b_names)
                            useful_sent.append(s.lower())
                            useful_label.append([rate1-pre_rate1])
                            # test_min.append(c_minutes)
                            break
                    elif j==' wonder':
                        if ('i wonder' not in s):
                            # s=replace_name(s,r_names,b_names)
                            useful_sent.append(s.lower())
                            useful_label.append([rate1-pre_rate1])
                            # test_min.append(c_minutes)
                            break  
                    else:
                        # s=replace_name(s,r_names,b_names)
                        useful_sent.append(s.lower())
                        useful_label.append([rate1-pre_rate1])
                        # test_min.append(c_minutes)
                        break


            
            p_minute=c_minutes
        p_time=c_time
        
    return useful_sent,useful_label
            
        
    #not ignore sentence before game start
    labels=[]

data=pd.read_csv('data.csv',converters = {"team_two_player1": str, "team_two_player2": str, "team_two_player3": str,"team_two_player4": str,"team_two_player5": str,"team_two_player6": str,
"team_two_player7": str,"team_two_player8": str,"team_two_player9": str,"team_two_player10": str,"team_one_name": str,"team_two_name": str})
inputs=[]
labels=[]
data=get_data(data)
for i in range(len(data)):
    j,l=preparedata(data.iloc[i])
    inputs+=j
    labels+=l
# with open("train_dict.pkl", "wb") as f:
#     pickle.dump({"text":inputs[:-20],'labels':labels[:-20]}, f)
# with open("test_dict.pkl", "wb") as f:
#     pickle.dump({"text":inputs[:-20],'labels':labels[:-20]}, f)
# with open("inter_partial_data_dict.pkl", "wb") as f:
#     pickle.dump({"text":inputs,'labels':labels}, f)
with open("norule_inter_partial.pkl", "wb") as f:
    pickle.dump({"text":inputs,'labels':labels}, f)
