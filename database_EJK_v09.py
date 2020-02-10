#===== 2020-02-06
#===== Crawling code for NIMS polymer database by Eunji Kim
#===== Version 09

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import time
import os

# Will create a folder (if it's not exit) to save output
def create_folder(dirName):
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass

# NIMS login info
def login_NIMS(NIMSid, NIMSpassword):
    url = 'http://polymer.nims.go.jp/PoLyInfo/cgi-bin/p-search.cgi'
    driver.get(url)
    username = driver.find_element_by_id("username")
    password = driver.find_element_by_id("password")
    username.send_keys(NIMSid)
    password.send_keys(NIMSpassword)
    driver.find_element_by_name("login").click()

    # >>
def manypages(soup2):
    nfind_text_match = soup2.find_all('input', {'type': 'submit', 'name': 'page',
                                                'style': 'font-size:8pt;width:3em;padding:0;', 'value': '>>'})
    if nfind_text_match:
        nfind = soup2.find_all('input',
                               {'type': 'submit', 'name': 'page', 'style': 'font-size:8pt;padding:0;'})
        npage = int(nfind[0]['value'].replace("(", "").replace(")", "").replace(" ", "").split('-')[-1])
    else:
        nfind = soup2.find_all('input', {'type': 'submit', 'name': 'page',
                                         'style': 'font-size:8pt;width:3em;padding:0;',
                                         'value': re.compile(r'[0-9]+')})
        # number of "link" pages : Duplicated
        npage = 1 + int(len(nfind) / 2)
    return npage

def pushmorebutton(i,soup2):
    if (i % 10 == 0):
        nfind_text_match = soup2.find_all('input', {'type': 'submit', 'name': 'page',
                                                    'style': 'font-size:8pt;width:3em;padding:0;', 'value': '>>'})
        if nfind_text_match:
            symbolpath = "//input[@type='submit' and @name='page'and @style='font-size:8pt;width:3em;padding:0;' and @value='>>']"
            driver.find_element_by_xpath(symbolpath)
        else:
            pass


#search polymer property
def pid_search(start, end):
    for polymer_id in range(start, end):
        # search
        time.sleep(3)
        search = driver.find_element_by_name('p-name-other')
        search.clear()
        polymer_key = 'P' + (str(polymer_id).zfill(6))
        print(polymer_key)
        search.send_keys(polymer_key)
        search.send_keys(Keys.RETURN)
        src = driver.page_source
        soup = BeautifulSoup(src, 'html.parser')
        empty_item = soup.find("td", {"class": "matches", "valign": "top"})
        match_res = empty_item.find('b').get_text()
        # Result of PID
        if re.match('Matches: not found in PoLyInfo Database.', match_res):
            driver.back()
        else:
            src1 = driver.page_source
            soup1 = BeautifulSoup(src1, 'html.parser')
            p_name = soup1.find('a', {'href': re.compile(r'/PoLyInfo/cgi-bin/pi-id-search')}).text
            prop = soup1.find_all('a', {'href': re.compile(r'/PoLyInfo/cgi-bin/ho-id-search')})
            prop_item = [p_item.text for p_item in prop]

            # property value
            if not prop_item:
                print('{} has no property'.format(polymer_key))
                driver.back()
            elif ('Glass transition temp.') not in prop_item:
                print("{} do not have Tg property".format(polymer_key))
                driver.back()
            else:
                # glass transition temp list index
                p_index = prop_item.index('Glass transition temp.')
                eachp = prop[p_index].text
                driver.find_element_by_link_text(eachp).click()
                src2 = driver.page_source
                soup2 = BeautifulSoup(src2, 'html.parser')
                table_head = soup2.find_all('td', {'class': 'header_small', 'align': re.compile(r'[a-z]+')})
                each_head = [head.text.replace('\n', '') for head in table_head]
                each_head[0] = polymer_key
                ncol = len(each_head)
                table_values = soup2.find_all('td', {'class': 'small_border', 'align': re.compile(r'[a-z]+')})
                each_value = [elem.text.replace('\n', '') for elem in table_values]
                output = pd.DataFrame(data=np.array(each_value).reshape(-1, ncol))
                npage = manypages(soup2)
                if npage > 1:
                    for i in range(2, npage + 1):
                        check = "@type='submit' and @name='page'and @style='font-size:8pt;width:3em;padding:0;' and @value="
                        add_c = "//input[" + str(check) + str(i) + str("]")
                        driver.find_element_by_xpath(add_c).send_keys(Keys.RETURN)
                        src2 = driver.page_source
                        soup2 = BeautifulSoup(src2, 'html.parser')
                        table_values = soup2.find_all('td', {'class': 'small_border', 'align': re.compile(r'[a-z]+')})
                        each_value = [elem.text.replace('\n', '') for elem in table_values]
                        # head and element to Series
                        new_values = pd.DataFrame(data=np.array(each_value).reshape(-1, ncol))
                        output = output.append(new_values)
                        pushmorebutton(i,soup2)

                        time.sleep(1)

                # save outputfile
                p_name = p_name.replace("/", "--")
                output.columns = [each_head]
                output.to_csv("P:/NIMS/{0}_{1}_tg.csv".format(polymer_key, p_name), index=False)
                # Use threshold cut-off npage = 40
                if npage > 20:
                    # There are 2 possible links with "Basics. We will use the first component.
                    # b_name = soup2.find('a', {'href': re.compile(r'/PoLyInfo/cgi-bin/p-search.cgi'+'\?s=basic')}).text
                    driver.find_element_by_link_text('Basic').click()
                else:
                    go_back = "window.history.go(-" + str(npage + 1) + ")"
                    driver.execute_script(go_back)



if __name__ == "__main__":

    create_folder("P:/NIMS")
    driver = webdriver.Chrome("C:/chromedriver.exe")
    login_NIMS("eun0116@kolon.com","0130asdf")
    ##### "For loop" for Polymer ID iteration
    startpid = 10001; endpid = 10010
    pid_search(startpid, endpid)
    print("Between {0:} and {1:} \n".format(startpid,endpid))
    driver.close()

