# ------------------------------------------------------------ #
#              _   _   _    _   _ _  ___ _    
#  ___ _ _ ___| |_| |_(_)__| |_| | || _ \ |   
# / _ \ '_/ _ \  _|  _| / _| / /_  _|   / |__ 
# \___/_| \___/\__|\__|_\__|_\_\ |_||_|_\____|
#---------------------------------------------
#Oregon Lottery - Pick 4 Predictor (w/ Randife)
#---------------------------------------------
#
#=============================================
#                  LINKS
#  ----------------------------------------
#
# + Kaggle: https://orottick4.randife.com/kaggle
#
# + GitHub: https://orottick4.randife.com/github
#
# + Lottery: https://orottick4.randife.com/lotte
#
#
#=============================================
#                 Copyright
#  ----------------------------------------
#
# "Oregon Lottery - Pick 4 Predictor (w/ Randife)" is written and copyrighted
# by Dinh Thoai Tran <dinhtt@randrise.com> [https://dinhtt.randrise.com]
#
#
#=============================================
#                 License
#  ----------------------------------------
#
# "Oregon Lottery - Pick 4 Predictor (w/ Randife)" is distributed under Apache-2.0 license
# [ https://github.com/dinhtt-randrise/orottick4rl/blob/main/LICENSE ]
#
# ------------------------------------------------------------ #

import random
import os
import json
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import pickle
import randife.randife as vrl

# ------------------------------------------------------------ #

class Orottick4RLRandomFormat(vrl.RandifeRandomFormat):
    def __init__(self, load_cache_dir = '/kaggle/working', save_cache_dir = '/kaggle/working'):
        super().__init__(1, [0], [9999], 0, 9999, -1, load_cache_dir, save_cache_dir)
        self.baseset = {0: 1000, 1: 100, 2: 10, 3: 1}

    def a2n(self, a):
        if a is None:
            return None
        if len(a) < 4:
            return None
        
        n = 0
        for ni in range(4):
            n += (a[ni]) * self.baseset[ni]
        return n
    
    def n2a(self, n):
        try:
            if n is None:
                return None

            a = []
            for ni in range(4):
                b = self.baseset[ni]
                c = int((n - (n % b)) // b)
                a.append(c)
                n = n - (c * b)

            return a
        except Exception as e:
            return None

    def get_pair_matching_keys(self):
        return ['m4', 'm3f', 'm3l', 'm3', 'm2']

    def match(self, win_rnd_num_list, prd_rnd_num_list, match_kind = 'ma'):
        if len(win_rnd_num_list) != self.get_size() or len(prd_rnd_num_list) != self.get_size():
            return False
        w = win_rnd_num_list[0]
        p = prd_rnd_num_list[0]
        if match_kind == 'ma':
            if w == p:
                return True
            else:
                return False
        elif match_kind == 'm4':
            if w == p:
                return True
            else:
                return False
        elif match_kind == 'm3f':
            wa = self.n2a(w)
            pa = self.n2a(p)
            if w < 0:
                return False
            elif wa[0] == pa[0] and wa[1] == pa[1] and wa[2] == pa[2]:
                return True
            else:
                return False
        elif match_kind == 'm3l':
            wa = self.n2a(w)
            pa = self.n2a(p)
            if w < 0:
                return False
            elif wa[1] == pa[1] and wa[2] == pa[2] and wa[3] == pa[3]:
                return True
            else:
                return False
        elif match_kind == 'm3':
            wa = self.n2a(w)
            pa = self.n2a(p)
            if w < 0:
                return False
            elif (wa[0] == pa[0] and wa[1] == pa[1] and wa[2] == pa[2]) or (wa[1] == pa[1] and wa[2] == pa[2] and wa[3] == pa[3]):
                return True
            else:
                return False
        elif match_kind == 'm2':
            wa = self.n2a(w)
            pa = self.n2a(p)
            if w < 0:
                return False
            elif (wa[0] == pa[0] and wa[1] == pa[1]) or (wa[1] == pa[1] and wa[2] == pa[2]) or (wa[2] == pa[2] and wa[3] == pa[3]):
                return True
            else:
                return False
        else:
            return False

    def heading(self, method, kind):
        if method == 'simulate':
            if kind == 'method_start':
                text = '''
=============================================
           ANALYZE SIMULATION
  ----------------------------------------
                '''
                return text
            if kind == 'method_end':
                text = '''
  ----------------------------------------
           ANALYZE SIMULATION
=============================================
                '''
                return text
            if kind == 'parameters_start':
                text = '''
  ----------------------------------------
               PARAMETERS
  ----------------------------------------
                '''
                return text
            if kind == 'parameters_end':
                text = '''
  ----------------------------------------
                '''
                return text
            if kind == 'prediction_start':
                text = '''
  ----------------------------------------
               PARAMETERS
  ----------------------------------------
                '''
                return text
            if kind == 'prediction_end':
                text = '''
  ----------------------------------------
                '''
                return text
        return ''
        
    def refine_json_pred(self, xdf, o_json_pred):
        x_time_no = xdf['time_no'].iloc[0]
        pdata = self.export_dict_time_data(x_time_no)
        date = pdata['date']
        buy_date = pdata['buy_date']
        next_date = pdata['next_date']
        sw = o_json_pred['w']
        lw = [int(x) for x in sw.split(', ')]
        w = lw[0]
        sn = o_json_pred['w']
        ln = [int(x) for x in sn.split(', ')]
        n = ln[0]
        sp = o_json_pred['pred']
        lp = sp.split(';')
        nlp = []
        m4 = 0
        m3f = 0
        m3l = 0
        m3 = 0
        m2 = 0
        xl_w = self.import_dataset_num_list(xdf, 0, 'w')
        for xs_p in lp:
            xs_p = xs_p.strip()
            if xs_p == '':
                continue
            xl_p = [int(x) for x in xs_p.split(', ')]
            if self.match(xl_w, xl_p, 'm4'):
                m4 += 1
            if self.match(xl_w, xl_p, 'm3f'):
                m3f += 1
            if self.match(xl_w, xl_p, 'm3l'):
                m3l += 1
            if self.match(xl_w, xl_p, 'm3'):
                m3 += 1
            if self.match(xl_w, xl_p, 'm2'):
                m2 += 1
            nlp.append(str(xl_p[0]))
        sp = ', '.join(nlp)
        m4_rsi = o_json_pred['ma_rsi']
        
        m4_sp = o_json_pred['ma_pred']
        m4_lp = m4_sp.split(';')
        m4_nlp = []
        for xs_p in m4_lp:
            xs_p = xs_p.strip()
            if xs_p == '':
                continue
            xl_p = [int(x) for x in xs_p.split(', ')]
            m4_nlp.append(str(xl_p[0]))
        m4_pred = ', '.join(m4_nlp)
        m4pc = o_json_pred['mapc']

        json_pred = {'time_no': int(o_json_pred['time_no']), 'date': date, 'buy_date': buy_date, 'next_date': next_date, 'w': int(w), 'n': int(n), 'sim_seed': int(o_json_pred['sim_seed']), 'date_cnt': int(o_json_pred['prc_time_cnt']), 'tck_cnt': int(o_json_pred['tck_cnt']), 'sim_cnt': o_json_pred['sim_cnt'], 'pred': sp, 'm4_rsi': int(m4_rsi), 'm4pc': int(m4pc), 'm4_pred': m4_pred, 'pcnt': int(1), 'm4': int(m4), 'm3f': int(m3f), 'm3l': int(m3l), 'm3': int(m3), 'm2': int(m2), 'm4_cnt': int(m4), 'm3f_cnt': int(m3f), 'm3l_cnt': int(m3l), 'm3': int(m3), 'm2': int(m2), 'mb_m4': int(0), 'mb_m3f': int(0), 'mb_m3l': int(0), 'mb_m3': int(0), 'mb_m2': int(0)}

        return json_pred

    def trim_dataset_sim_input(self, ddf):
        cols = ['time_no']
        for mix in range(self.size):
            no = mix + 1
            cols.append(f'w_{no}')
        for mix in range(self.size):
            no = mix + 1
            cols.append(f'n_{no}')
        cols.append('sim_seed')
        cols.append('sim_cnt')
        return ddf[cols]

    def capture_map(self, pdf, x_sim_seed):
        return self.capture_m4p(pdf, x_sim_seed)

    def capture_m4p_p_1(self, pdf):
        xdf = pdf.sort_values(by=['time_no'], ascending=[False])
        xdf = xdf[(xdf['m4'] == 0)&(xdf['m3f'] == 0)&(xdf['m3l'] == 0)&(xdf['m3'] == 0)&(xdf['m2'] == 1)&(xdf['a_m4'] == 0)&(xdf['a_m3f'] == 0)&(xdf['a_m3l'] == 0)&(xdf['a_m3'] == 0)&(xdf['a_m2'] > 0)&(xdf['m4_cnt'] > 0)&(xdf['m4_cnt'] < 10)]
        if len(xdf) == 0:
            return None
        else:
            for ri in range(len(xdf)):
                if xdf['a_m2'].iloc[ri] * 2 == xdf['m4_cnt'].iloc[ri]:
                    time_no = xdf['time_no'].iloc[ri]
                    zdf = pdf[pdf['time_no'] > time_no]
                    if len(zdf) <= 60 and len(zdf) > 30:
                        return time_no
                    else:
                        return None
            return None

    def capture_m4p_p_2(self, pdf):
        xdf = pdf.sort_values(by=['time_no'], ascending=[False])
        xdf1 = xdf[(xdf['m4'] == 0)&(xdf['m3f'] == 0)&(xdf['m3l'] == 0)&(xdf['m2'] == 0)&(xdf['a_m4'] == 0)&(xdf['a_m3f'] == 1)&(xdf['a_m3l'] == 0)]
        if len(xdf1) == 0:
            return None
        else:
            time_no = xdf1['time_no'].iloc[0]
            xdf2 = xdf[xdf['time_no'] == time_no]
            if len(xdf2) == 0:
                return None
            else:
                time_no = xdf2['time_no'].iloc[0]
                zdf = pdf[pdf['time_no'] > time_no]
                if len(zdf) <= 28 and len(zdf) > 14:
                    return time_no
                else:
                    return None

    def capture_m4p_p_3(self, pdf):
        xdf = pdf.sort_values(by=['time_no'], ascending=[False])
        xdf1 = xdf[(xdf['m4'] == 0)&(xdf['m3f'] == 0)&(xdf['m3l'] == 0)&(xdf['m3'] == 0)&(xdf['m2'] == 1)&(xdf['a_m2'] > 0)&(xdf['m4_cnt'] > 0)]
        if len(xdf1) == 0:
            return None
        else:
            xdf2 = xdf[xdf['m4'] == 1]
            if len(xdf2) == 0:
                return None
            m4_time_no = xdf2['time_no'].iloc[0]
            xdf1 = xdf1.sort_values(by=['time_no'], ascending=[False])
            for ri in range(len(xdf1)):
                time_no = xdf1['time_no'].iloc[ri]
                a = xdf1['a_m2'].iloc[ri]
                b = 2 * (xdf1['m4_cnt'].iloc[ri] / 3)
                if a == b and time_no < m4_time_no: 
                    zdf = pdf[pdf['time_no'] > time_no]
                    if len(zdf) <= 36 and len(zdf) > 18:
                        return time_no
                    else:
                        return None
            return None
            
    def join_m4p(self, pdf, adf, l_time_no, time_no):
        if time_no is not None:
            if time_no not in l_time_no:
                l_time_no.append(time_no)
                df = pdf[pdf['time_no'] == time_no]
                df['m4p_no'] = len(l_time_no)
                if adf is None:
                    adf = df
                else:
                    adf = pd.concat([adf, df])
                adf = adf.sort_values(by=['m4p_no'], ascending=[True])
        return l_time_no, adf
        
    def capture_m4p(self, pdf, x_sim_seed):
        l_pred = []
        l_time_no = []
        adf = None

        time_no = self.capture_m4p_p_1(pdf)
        l_time_no, adf = self.join_m4p(pdf, adf, l_time_no, time_no)

        sz = 0
        if adf is not None:
            sz = len(adf)
        print(f'=> [M4PC-1] {l_time_no} -> {sz}')

        time_no = self.capture_m4p_p_2(pdf)
        l_time_no, adf = self.join_m4p(pdf, adf, l_time_no, time_no)

        sz = 0
        if adf is not None:
            sz = len(adf)
        print(f'=> [M4PC-2] {l_time_no} -> {sz}')

        time_no = self.capture_m4p_p_3(pdf)
        l_time_no, adf = self.join_m4p(pdf, adf, l_time_no, time_no)

        sz = 0
        if adf is not None:
            sz = len(adf)
        print(f'=> [M4PC-3] {l_time_no} -> {sz}')
        
        if adf is None or len(l_time_no) == 0 or len(adf) == 0:
            return l_pred

        adf = adf.sort_values(by=['m4p_no'], ascending=[True])
        for ri in range(len(adf)):
            x_sim_cnt = adf['sim_cnt'].iloc[ri]
            p = self.reproduce(x_sim_seed, x_sim_cnt)
            l_pred.append(p)

        return l_pred

class Orottick4RLSimulator:
    def __init__(self, prd_sort_order = 'A', has_step_log = True, m4p_obs = False, m4p_cnt = -1, m4p_vry = True, load_cache_dir = '/kaggle/working', save_cache_dir = '/kaggle/working', heading_printed = False):
        self.heading_printed = heading_printed

        if prd_sort_order not in ['A', 'B', 'C']:
            prd_sort_order = 'A'
        self.prd_sort_order = prd_sort_order

        self.has_step_log = has_step_log
        self.m4p_obs = m4p_obs
        self.m4p_cnt = m4p_cnt

        if m4p_cnt < 2:
            m4p_vry = False
        self.m4p_vry = m4p_vry

        self.load_cache_dir = load_cache_dir
        self.save_cache_dir = save_cache_dir

        self.rnd_format = Orottick4RLRandomFormat(self.load_cache_dir, self.save_cache_dir)

    def print_heading(self):
        if self.heading_printed:
            return
        self.heading_printed = True
            
        text = '''
              _   _   _    _   _ _  ___ _    
  ___ _ _ ___| |_| |_(_)__| |_| | || _ \ |   
 / _ \ '_/ _ \  _|  _| / _| / /_  _|   / |__ 
 \___/_| \___/\__|\__|_\__|_\_\ |_||_|_\____|
---------------------------------------------
Oregon Lottery - Pick 4 Predictor (w/ Randife)
---------------------------------------------
        '''
        print(text)        

    def get_time_no(self, draw_date):
        time_forced = '00.00.00'
        mark_time = datetime.strptime('2000.01.01' + '.' + time_forced, "%Y.%m.%d.%H.%M.%S")
        last_time = datetime.strptime(draw_date + '.' + time_forced, "%Y.%m.%d.%H.%M.%S")
        dd = last_time - mark_time
        ds = dd.total_seconds()
        days = int(round(ds / (60 * 60 * 24)))
        return days
        
    def download_drawing(self, buffer_dir, lotte_kind, v_date):
        self.print_heading()

        text = '''
=============================================
             DOWNLOAD DRAWING
  ----------------------------------------
        '''
        print(text) 

        text = '''
  ----------------------------------------
               PARAMETERS
  ----------------------------------------
        '''
        print(text) 

        print(f'[BUFFER_DIR] {buffer_dir}')
        print(f'[LOTTE_KIND] {lotte_kind}')
        print(f'[DATE] {v_date}')

        text = '''
  ----------------------------------------
        '''
        print(text) 

        tmp_dir = buffer_dir
        work_dir = f'{tmp_dir}/data-' + str(random.randint(1, 1000000))  
        os.system(f'mkdir -p {work_dir}')

        curl_file = f'{work_dir}/curl.txt'
        fds = v_date.split('.')
        END_DATE = str(int(fds[1])) + '/' + str(int(fds[2])) + '/' + str(int(fds[0]))[2:]
        cmd = "curl 'https://api2.oregonlottery.org/drawresults/ByDrawDate?gameSelector=p4&startingDate=01/01/1984&endingDate=" + END_DATE + "&pageSize=50000&includeOpen=False' -H 'Accept: application/json, text/javascript, */*; q=0.01' -H 'Accept-Language: en-US,en;q=0.9' -H 'Cache-Control: no-cache' -H 'Connection: keep-alive' -H 'Ocp-Apim-Subscription-Key: 683ab88d339c4b22b2b276e3c2713809' -H 'Origin: https://www.oregonlottery.org' -H 'Pragma: no-cache' -H 'Referer: https://www.oregonlottery.org/' -H 'Sec-Fetch-Dest: empty' -H 'Sec-Fetch-Mode: cors' -H 'Sec-Fetch-Site: same-site' -H 'User-Agent: Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36' -H 'sec-ch-ua: " + '"' + "Not/A)Brand" + '"' + ";v=" + '"' + "8" + '"' + ", " + '"' + "Chromium" + '"' + ";v=" + '"' + "126" + '"' + ", " + '"' + "Google Chrome" + '"' + ";v=" + '"' + "126" + '"' + "' -H 'sec-ch-ua-mobile: ?0' -H 'sec-ch-ua-platform: " + '"' + "Chrome OS" + '"' + "' > " + curl_file
        print(cmd)
        os.system(cmd)

        T = 'T13:00:00'
        if lotte_kind == 'p4a':
            T = 'T13:00:00'
        if lotte_kind == 'p4b':
            T = 'T16:00:00'
        if lotte_kind == 'p4c':
            T = 'T19:00:00'
        if lotte_kind == 'p4d':
            T = 'T22:00:00'
            
        FIND = v_date.replace('.', '-') + T
        data = []
        if os.path.exists(curl_file):
            with open(curl_file, 'r') as f:
                data = json.load(f)
        line = ''
        for di in range(len(data)):
            et = data[di]
            if et['DrawDateTime'] == FIND:
                line = 'yes'
                break
        
        if line == '':
            os.system(f'rm -rf {work_dir}')
            print(f'== [Error] ==> Drawing is not found!')
            return None

        rows = []
        for di in range(len(data)):
            et = data[di]
            if T not in et['DrawDateTime']:
                continue
            date = et['DrawDateTime'].split('T')[0].replace('-', '.')  
            sl = et['WinningNumbers']
            n = (sl[0] * 1000) + (sl[1] * 100) + (sl[2] * 10) + (sl[3] * 1)
            rw = {'date': date, 'w': -1, 'n': n}
            rows.append(rw)
        ddf = pd.DataFrame(rows)    

        rows = []
        date_list = ddf['date'].unique()
        for today in date_list:
            d1 = datetime.strptime(today, "%Y.%m.%d")
            d2 = d1 + timedelta(minutes=int(+(1) * (60 * 24)))
            buy_date = d2.strftime('%Y.%m.%d')   
            tdf = ddf[ddf['date'] == today]
            bdf = ddf[ddf['date'] == buy_date]
            next_date = buy_date
            n = tdf['n'].iloc[0]
            if len(bdf) == 0:
                w = -1
            else:
                w = bdf['n'].iloc[0]
            time_no = self.get_time_no(today)
            rw = {'date': today, 'buy_date': buy_date, 'next_date': next_date, 'time_no': time_no, 'w_1': w, 'n_1': n, 'sim_seed': -1, 'sim_cnt': -1}
            rows.append(rw)
        df = pd.DataFrame(rows)
        df = df.sort_values(by=['date'], ascending=[False])
        os.system(f'rm -rf {work_dir}')

        sz = len(df)
        print(f'== [Success] ==> Drawing data is downloaded. It contains {sz} rows.')
        
        text = '''
  ----------------------------------------
             DOWNLOAD DRAWING
=============================================
        '''
        print(text) 

        return df
        
    def simulate(self, v_buy_date, buffer_dir = '/kaggle/buffers/orottick4', lotte_kind = 'p4a', data_df = None, v_date_cnt = 56, tck_cnt = 2, runtime = None, cache_only = False):
        self.print_heading()

        if data_df is None:
            d1 = datetime.strptime(v_buy_date, "%Y.%m.%d")
            g = -1
            d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
            v_date = d2.strftime('%Y.%m.%d')
    
            data_df = self.download_drawing(buffer_dir, lotte_kind, v_date)
            if data_df is None:
                return None, None, None

        xdf = data_df[data_df['buy_date'] == v_buy_date]
        prd_time_no = xdf['time_no'].iloc[0]

        self.rnd_format.reset_dict_time_data()
        for rwi in range(len(data_df)):
            z_time_no = data_df['time_no'].iloc[rwi]
            z_date = data_df['date'].iloc[rwi]
            z_buy_date = data_df['buy_date'].iloc[rwi]
            z_next_date = data_df['next_date'].iloc[rwi]
            z_data = {'date': z_date, 'buy_date': z_buy_date, 'next_date': z_next_date}
            self.rnd_format.import_dict_time_data(z_time_no, z_data)
            
        prd_sim = self.rnd_format.create_random_simulator()
        sdf, mdf, pdf, json_pred, n_json_pred = prd_sim.simulate(data_df, prd_time_no, v_date_cnt, runtime, tck_cnt, self.m4p_cnt, self.has_step_log, cache_only)

        if mdf is not None:
            mdf['date'] = ''
            mdf['buy_date'] = ''
            mdf['next_date'] = ''
            for rwi in range(len(mdf)):
                z_time_no = mdf[f'time_no'].iloc[rwi]
                z_data = self.rnd_format.export_dict_time_data(z_time_no)
                self.rnd_format.export_dataset_num(mdf, rwi, f'date', z_data['date'])
                self.rnd_format.export_dataset_num(mdf, rwi, f'buy_date', z_data['buy_date'])
                self.rnd_format.export_dataset_num(mdf, rwi, f'next_date', z_data['next_date'])

        if pdf is not None:
            pdf['date'] = ''
            pdf['buy_date'] = ''
            pdf['next_date'] = ''
            for rwi in range(len(pdf)):
                z_time_no = pdf[f'time_no'].iloc[rwi]
                z_data = self.rnd_format.export_dict_time_data(z_time_no)
                self.rnd_format.export_dataset_num(pdf, rwi, f'date', z_data['date'])
                self.rnd_format.export_dataset_num(pdf, rwi, f'buy_date', z_data['buy_date'])
                self.rnd_format.export_dataset_num(pdf, rwi, f'next_date', z_data['next_date'])
        
        return sdf, mdf, pdf, n_json_pred

    def observe(self, lotte_kind, v_buy_date, o_max_tck = 2, o_date_cnt = 56, o_runtime = 60 * 60 * 11.5, date_cnt = 56, buffer_dir = '/kaggle/buffers/orottick4', data_df = None, cache_only = False):
        self.print_heading()

        start_time = time.time()

        more = {}
        
        text = '''
=============================================
                 OBSERVE
  ----------------------------------------
        '''
        print(text)

        text = '''
  ----------------------------------------
                PARAMETERS
  ----------------------------------------
        '''
        print(text) 

        v_data_df_is_none = False
        if data_df is None:
            v_data_df_is_none = True
            
        print(f'[BUFFER_DIR] {buffer_dir}')
        print(f'[LOTTE_KIND] {lotte_kind}')
        print(f'[DATA_DF_IS_NONE] {v_data_df_is_none}')
        print(f'[BUY_DATE] {v_buy_date}')
        print(f'[DATE_CNT] {date_cnt}')
        print(f'[O_DATE_CNT] {o_date_cnt}')
        print(f'[TCK_CNT] {o_max_tck}')
        print(f'[RUNTIME] {o_runtime}')
        rs = 'yes' if cache_only else 'no'
        print(f'[CACHE_ONLY] {rs}')

        text = '''
  ----------------------------------------
        '''
        print(text) 

        d1 = datetime.strptime(v_buy_date, "%Y.%m.%d")
        g = -1
        d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
        v_date = d2.strftime('%Y.%m.%d')

        if data_df is None:
            data_df = self.download_drawing(buffer_dir, lotte_kind, v_date)
            
        if data_df is None:
            return None, more

        ddf = data_df[data_df['buy_date'] < v_buy_date]
        ddf = ddf.sort_values(by=['buy_date'], ascending=[False])
        ddf = ddf[:o_date_cnt]
        ddf = ddf.sort_values(by=['buy_date'], ascending=[True])

        tck_cnt = 0
        pcnt = 0
        m4_cnt = 0
        m3f_cnt = 0
        m3l_cnt = 0
        m3_cnt = 0
        m2_cnt = 0
        rows = []
        for ri in range(len(ddf)):
            if o_runtime is not None:
                if time.time() - start_time > o_runtime:
                    break
                
            t_date = ddf['date'].iloc[ri]
            t_buy_date = ddf['buy_date'].iloc[ri]
            t_next_date = ddf['next_date'].iloc[ri]
            t_w = ddf['w_1'].iloc[ri]
            t_n = ddf['n_1'].iloc[ri]

            text = '''
  ----------------------------------------
  [O] __DATE__ : __W__
  ----------------------------------------
        '''
            print(text.replace('__DATE__', str(t_buy_date)).replace('__W__', str(t_w))) 

            runtime = None
            if o_runtime is not None:
                o_overtime = time.time() - start_time
                runtime = o_runtime - o_overtime
            sdf, mdf, pdf, json_prd = self.simulate(t_buy_date, buffer_dir, lotte_kind, data_df, date_cnt, o_max_tck, runtime, cache_only)
            if sdf is None or mdf is None or json_prd is None or pdf is None:
                continue
            if cache_only:
                continue
                
            more[f'pred_{t_buy_date}'] = json_prd
            more[f'sim_{t_buy_date}'] = sdf
            more[f'pick_{t_buy_date}'] = pdf
            more[f'match_{t_buy_date}'] = mdf
            
            t_pred = json_prd['pred']
            vry = True
            if self.m4p_obs:
                t_pred = json_prd['m4_pred']
                if self.m4p_vry:
                    if len(t_pred.split(', ')) != 1:
                        vry = False
            t_prd_lst = t_pred.split(', ')
            if o_max_tck > 0:
                if len(t_prd_lst) > o_max_tck:
                    t_prd_lst = t_prd_lst[:o_max_tck]
            nlst = [int(x) for x in t_prd_lst]
            pcnt += 1
            prd_num = len(nlst)
            tck_cnt += prd_num
            m4 = 0
            m3f = 0
            m3l = 0
            m3 = 0
            m2 = 0
            for t_p in nlst:
                if vry and self.rnd_format.match([t_w], [t_p], 'm4'):
                    m4 += 1
                if vry and self.rnd_format.match([t_w], [t_p], 'm3f'):
                    m3f += 1
                if vry and self.rnd_format.match([t_w], [t_p], 'm3l'):
                    m3l += 1
                if vry and self.rnd_format.match([t_w], [t_p], 'm3'):
                    m3 += 1
                if vry and self.rnd_format.match([t_w], [t_p], 'm2'):
                    m2 += 1
            m4_cnt += m4
            m3f_cnt += m3f
            m3l_cnt += m3l
            m3_cnt += m3
            m2_cnt += m2

            rw = json_prd
            rw['m4'] = m4
            rw['m3f'] = m3f
            rw['m3l'] = m3l
            rw['m3'] = m3
            rw['m2'] = m2
            rw['m4_cnt'] = m4_cnt
            rw['m3f_cnt'] = m3f_cnt
            rw['m3l_cnt'] = m3l_cnt
            rw['m3_cnt'] = m3_cnt
            rw['m2_cnt'] = m2_cnt
            rw['pcnt'] = pcnt
            
            rows.append(rw)

            print(str(rw))

        odf = None

        if len(rows) > 0:
            odf = pd.DataFrame(rows)
            odf = odf.sort_values(by=['buy_date'], ascending=[False])
        
        text = '''
  ----------------------------------------
                 OBSERVE
=============================================
        '''
        print(text)

        return odf, more

    def get_option(options, key, def_val):
        if key in options:
            return options[key]
        return def_val

    def run(options, github_pkg, non_github_create_fn = None):
        BUY_DATE = Orottick4RLSimulator.get_option(options, 'BUY_DATE', '2025.03.27')
        BUFFER_DIR = Orottick4RLSimulator.get_option(options, 'BUFFER_DIR', '/kaggle/buffers/orottick4')
        LOTTE_KIND = Orottick4RLSimulator.get_option(options, 'LOTTE_KIND', 'p4a')
        DATA_DF = Orottick4RLSimulator.get_option(options, 'DATA_DF', None)
        DATE_CNT = Orottick4RLSimulator.get_option(options, 'DATE_CNT', 56 * 5)
        O_DATE_CNT = Orottick4RLSimulator.get_option(options, 'O_DATE_CNT', 7)
        TCK_CNT = Orottick4RLSimulator.get_option(options, 'TCK_CNT', 56 * 5)
        F_TCK_CNT = Orottick4RLSimulator.get_option(options, 'F_TCK_CNT', 250)
        RUNTIME = Orottick4RLSimulator.get_option(options, 'RUNTIME', 60 * 60 * 11.5)
        PRD_SORT_ORDER = Orottick4RLSimulator.get_option(options, 'PRD_SORT_ORDER', 'B')
        HAS_STEP_LOG = Orottick4RLSimulator.get_option(options, 'HAS_STEP_LOG', True)
        RANGE_CNT = Orottick4RLSimulator.get_option(options, 'RANGE_CNT', 52)
        M4P_OBS = Orottick4RLSimulator.get_option(options, 'M4P_OBS', False)
        M4P_CNT = Orottick4RLSimulator.get_option(options, 'M4P_CNT', 3)
        M4P_VRY = Orottick4RLSimulator.get_option(options, 'M4P_VRY', False)
        M4P_ONE = Orottick4RLSimulator.get_option(options, 'M4P_ONE', True)
        RESULT_DIR = Orottick4RLSimulator.get_option(options, 'RESULT_DIR', '/kaggle/working')
        LOAD_CACHE_DIR = Orottick4RLSimulator.get_option(options, 'LOAD_CACHE_DIR', '/kaggle/working')
        SAVE_CACHE_DIR = Orottick4RLSimulator.get_option(options, 'SAVE_CACHE_DIR', '/kaggle/working')
        CACHE_CNT = Orottick4RLSimulator.get_option(options, 'CACHE_CNT', -1)
        CACHE_ONLY = Orottick4RLSimulator.get_option(options, 'CACHE_ONLY', False)
        USE_GITHUB = Orottick4RLSimulator.get_option(options, 'USE_GITHUB', False)
        METHOD = Orottick4RLSimulator.get_option(options, 'METHOD', 'simulate')

        if non_github_create_fn is None:
            USE_GITHUB = True
            
        if USE_GITHUB:
            ok4s = github_pkg.Orottick4RLSimulator(PRD_SORT_ORDER, HAS_STEP_LOG, M4P_OBS, M4P_CNT, M4P_VRY, LOAD_CACHE_DIR, SAVE_CACHE_DIR)
        else:
            ok4s = non_github_create_fn(PRD_SORT_ORDER, HAS_STEP_LOG, M4P_OBS, M4P_CNT, M4P_VRY, LOAD_CACHE_DIR, SAVE_CACHE_DIR)
                
        if METHOD == 'simulate':
            sdf, mdf, pdf, json_pred = ok4s.simulate(BUY_DATE, BUFFER_DIR, LOTTE_KIND, DATA_DF, DATE_CNT, TCK_CNT, RUNTIME, CACHE_ONLY)
            if sdf is not None:
                sdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{BUY_DATE}.csv', index=False)
            if mdf is not None:
                mdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-match-{BUY_DATE}.csv', index=False)
            if pdf is not None:
                pdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-pick-{BUY_DATE}.csv', index=False)
            if json_pred is not None:
                with open(f'{RESULT_DIR}/{LOTTE_KIND}-pred-{BUY_DATE}.json', 'w') as f:
                    json.dump(json_pred, f)
        
                text = '''
====================================
    PREDICT: [__LK__] __BD__
  -------------------------------

+ Predicted Numbers: __RS__

+ M4P Numbers: __M4__

+ Win Number:

+ Result: 

+ M4P Result:

+ M4PC: __M4PC__

+ Predict Notebook:


  -------------------------------
              MONEY
  -------------------------------

+ Period No: 

+ Day No: 

+ Tickets: 250

+ Cost: $325

+ Total Cost: $325

+ Broker Cost: $0

+ Total Broker Cost: $0

+ Prize: $0

+ Total Prize: $0

+ Current ROI: 0.0%

+ Current ROI (w/o broker): 0.0%


  -------------------------------
            REAL BUY
  -------------------------------

+ Buy Number: __M4__

+ Confirmation Number: 

+ Cost: $1.3

+ Total Cost: $1.3

+ Broker Cost: $0.3

+ Total Broker Cost: $0.3

+ Prize: $0

+ Total Prize: $0

+ Current ROI: 0.0%

+ Current ROI (w/o broker): 0.0%
        
                '''
                if F_TCK_CNT != TCK_CNT:
                    lx_pred = str(json_pred['pred']).split(', ')
                    if len(lx_pred) > F_TCK_CNT:
                        lx_pred = lx_pred[:F_TCK_CNT]
                    json_pred['pred'] = ', '.join(lx_pred)
                m4pc = json_pred['m4pc']
                if m4pc > 0:
                    if M4P_ONE:
                        lx = str(json_pred['m4_pred']).split(', ')
                        if len(lx) != 1:
                            m4pc = 0
                json_pred['m4pc'] = m4pc
                text = text.replace('__LK__', str(LOTTE_KIND)).replace('__BD__', str(BUY_DATE)).replace('__RS__', str(json_pred['pred'])).replace('__M4__', str(json_pred['m4_pred'])).replace('__M4PC__', str(json_pred['m4pc']))
                with open(f'{RESULT_DIR}/{LOTTE_KIND}-pred-{BUY_DATE}.txt', 'w') as f:
                    f.write(text)
                print(text)
        
        if METHOD == 'observe':
            odf, more = ok4s.observe(LOTTE_KIND, BUY_DATE, TCK_CNT, O_DATE_CNT, RUNTIME, DATE_CNT, BUFFER_DIR, DATA_DF, CACHE_ONLY)
        
            if odf is not None and more is not None and len(odf) > 0:
                odf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-observe-{BUY_DATE}.csv', index=False)
                qdf = odf[odf['m4'] > 0]
                if len(qdf) > 0:
                    for ri in range(len(qdf)):
                        t_buy_date = qdf['buy_date'].iloc[ri]
        
                        key = 'pred_' + t_buy_date    
                        if key in more:
                            json_pred = more[key]
                            if json_pred is not None:
                                with open(f'{RESULT_DIR}/{LOTTE_KIND}-pred-{t_buy_date}.json', 'w') as f:
                                    json.dump(json_pred, f)
        
                        key = 'sim_' + t_buy_date                
                        if key in more:
                            xdf = more[key]
                            if xdf is not None:
                                xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)

                        key = 'match_' + t_buy_date                
                        if key in more:
                            xdf = more[key]
                            if xdf is not None:
                                xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)

                        key = 'pick_' + t_buy_date                
                        if key in more:
                            xdf = more[key]
                            if xdf is not None:
                                xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-pick-{t_buy_date}.csv', index=False)

                qdf = odf[odf['m4'] <= 0]
                if len(qdf) > 0:
                    for ri in range(len(qdf)):
                        t_buy_date = qdf['buy_date'].iloc[ri]
                
                        key = 'match_' + t_buy_date                
                        if key in more:
                            xdf = more[key]
                            if xdf is not None:
                                xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)

                        key = 'pick_' + t_buy_date                
                        if key in more:
                            xdf = more[key]
                            if xdf is not None:
                                xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-pick-{t_buy_date}.csv', index=False)
                
        if METHOD == 'observe_range':
            start_time = time.time()
            range_idx = 0
            while range_idx < RANGE_CNT:
                if time.time() - start_time > RUNTIME:
                    break
                    
                d1 = datetime.strptime(BUY_DATE, "%Y.%m.%d")
                g = -(range_idx * O_DATE_CNT)
                d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
                v_buy_date = d2.strftime('%Y.%m.%d')
                o_overtime = time.time() - start_time
                v_runtime = RUNTIME - o_overtime
        
                odf, more = ok4s.observe(LOTTE_KIND, v_buy_date, TCK_CNT, O_DATE_CNT, v_runtime, DATE_CNT, BUFFER_DIR, DATA_DF, CACHE_ONLY)
            
                if odf is not None and more is not None and len(odf) > 0:
                    odf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-observe-{v_buy_date}.csv', index=False)

                    qdf = odf[odf['m4'] > 0]
                    if len(qdf) > 0:
                        for ri in range(len(qdf)):
                            t_buy_date = qdf['buy_date'].iloc[ri]
            
                            key = 'pred_' + t_buy_date    
                            if key in more:
                                json_pred = more[key]
                                if json_pred is not None:
                                    with open(f'{RESULT_DIR}/{LOTTE_KIND}-pred-{t_buy_date}.json', 'w') as f:
                                        json.dump(json_pred, f)
            
                            key = 'sim_' + t_buy_date                
                            if key in more:
                                xdf = more[key]
                                if xdf is not None:
                                    xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)
    
                            key = 'match_' + t_buy_date                
                            if key in more:
                                xdf = more[key]
                                if xdf is not None:
                                    xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)
    
                            key = 'pick_' + t_buy_date                
                            if key in more:
                                xdf = more[key]
                                if xdf is not None:
                                    xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-pick-{t_buy_date}.csv', index=False)
    
                    qdf = odf[odf['m4'] <= 0]
                    if len(qdf) > 0:
                        for ri in range(len(qdf)):
                            t_buy_date = qdf['buy_date'].iloc[ri]
                    
                            key = 'match_' + t_buy_date                
                            if key in more:
                                xdf = more[key]
                                if xdf is not None:
                                    xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)
    
                            key = 'pick_' + t_buy_date                
                            if key in more:
                                xdf = more[key]
                                if xdf is not None:
                                    xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-pick-{t_buy_date}.csv', index=False)                    
        
                range_idx += 1
        
        if METHOD == 'download':  
            d1 = datetime.strptime(BUY_DATE, "%Y.%m.%d")
            g = -1
            d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
            v_date = d2.strftime('%Y.%m.%d')
        
            data_df = ok4s.download_drawing(BUFFER_DIR, LOTTE_KIND, v_date)
        
            if data_df is not None:
                data_df.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-{BUY_DATE}.csv', index=False)

        
# ------------------------------------------------------------ #
