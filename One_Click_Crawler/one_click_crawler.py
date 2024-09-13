'''
gpt-3.5-turbo
gpt-4
claude-3-haiku-20240307
claude-3-sonnet-20240229
claude-3-opus-20240229
gemini-pro
gemini-1.0-pro-001
gemini-1.5-pro
gemini-pro-version
gemini-1.0-pro-vision-001
'''

import time
import json
import os
import re
import uuid
import openai
import http.client
import yaml
import argparse

import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

from datetime import datetime,timedelta
from log_config import configure_logging
import settings


class Search:
    config_path = '/home/zhouzihao/one_click_crawler/configs/one_click_crawler.yaml'
    api_base = ''
    api_key = '' 

    request_record = []
    question_list = []
    
    query_count = 20 
    query_prefix = f"Please answer my question strictly according to the following template, no extra text is needed except for the answer :\n\n 1.answer1 \n 2.answer2 \n...\nI need QUERY_COUNT answers of the question. The question is: \n"

    # 一共有三个步骤：query、search、crawl
    # 下面三个变量判断本次任务从哪一步开始
    step_list = []

    search_domain = 'api.sohoyo.io'
    search_query = ''
    search_model = 'google-search'
    search_authorization = ''
    search_content_type = 'application/json; charset=utf-8'
    search_sub_url = '/v1/completions'
    search_encode = 'utf-8'

    search_record = []

    # crawl相关的参数
    crawl_urls = []
    # crawl_search_index>=0，爬取开始的index
    crawl_search_index = 0
    # 是否只爬取一个批次，即当request_id下的crawl_search_index爬取完之后就结束，默认为False，除了当前还会寻找之后的
    crawl_single_batch = False

    crawl_save_types = ['html', 'img', 'pdf']

    # reques_id / search_index / crawl_time / from_url/ content_type / file_dir / file_name / file_type / file_size  
    crawl_record = []
    crawl_timeout = 60
    crawl_text_threshold = 200

        
    # 判断step的规则是否正确
    def __init__(self) :
        # 创建 ArgumentParser 对象
        parser = argparse.ArgumentParser(description='这是一个命令行参数解析器')

        # 添加一个可选的命令行参数'--config'
        parser.add_argument('--config', type=str, help='配置文件路径')

        # 解析命令行参数
        args = parser.parse_args()
        if args.config:
            self.config_path = args.config
            print(f"配置文件路径：{args.config}")
        else:
            print(f"使用默认配置文件：{self.config_path}")

    def change_size(size_num):
        index = 0
        unit = ['B', 'KB', 'MB', 'GB', 'TB']
        while size_num > 1024:
            size_num = round(size_num / 1024, 2)
            index += 1
        return f'{size_num}{unit[index]}'

    def right_order(self, step_list):
        query_indexes = [index for index, step in enumerate(step_list) if step == 'query']
        search_indexes = [index for index, step in enumerate(step_list) if step == 'search']
        crawl_indexes = [index for index, step in enumerate(step_list) if step == 'crawl']

        query_num = len(query_indexes)
        search_num = len(search_indexes)
        crawl_num = len(crawl_indexes)
        total_num = query_num + search_num + crawl_num

        if query_num > 1 or search_num > 1 or crawl_num > 1:
            raise StepError(step_list) 
        if total_num == 0 or total_num > 3:
            raise StepError(step_list) 

        query_index = query_indexes[0] if query_num == 1 else -1
        search_index = search_indexes[0] if search_num == 1 else -1
        crawl_index = crawl_indexes[0] if crawl_num == 1 else -1
        if query_index > search_index or query_index > crawl_index or search_index > crawl_index:
            raise StepError(step_list) 
        
        return True

    def load_config(self):
        # setting： 设置必须要添加的参数，相当于api调用规则
        # config： 从yaml中读取的配置
        with open(self.config_path, 'r') as cp:
            config = yaml.safe_load(cp)
        # 先建立日志
            log_dir = config.get('log_dir')
            logger_name = config.get('logger_name')
                    
            if not log_dir:
                raise ConfigLoadError('log_dir')
            if not logger_name:
                raise ConfigLoadError('logger_name')
            
            current_time = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
            log_path = f'{log_dir}/{logger_name}_{current_time}.log'
            self.logger = configure_logging(log_path, logger_name)

            step_list = config.get('step_list')
            self.step_list = step_list
            if not step_list:
                raise ConfigLoadError('step_list')
            
            for step in step_list:
                step_config = config.get(step)
                if not step_config:
                    raise ConfigLoadError(step)
                else:
                    self.add_properties(config, step)

    def add_properties(self, config, type):
        setting = settings.settings[type]
        must_setting = setting['must']
        optional_setting = setting['optional']

        config = config.get(type)

        # 读入必须配置
        for key in must_setting.keys():
            value = config.get(key)
            if not value:
                self.logger.error(f'配置文件不对，参数 {key} 缺失！！')
                return                   
            # 动态给实例赋值
            setattr(self, key, value)

        # 读入非必须配置
        for key in optional_setting.keys():
            value = config.get(key)
            if value:
                setattr(self, key, value)
      

    def do_query(self):
        
        try:
            begin_time = datetime.now()
            begin_time_str = begin_time.strftime('%Y_%m_%d_%H:%M:%S')
            openai.api_base = self.api_base
            openai.api_key = self.api_key
            self.query_prefix = self.query_prefix.replace('QUERY_COUNT', str(self.query_count), 1)
            self.logger.info(f'开始执行search，choose_model为:: {self.choose_model}, query为:: {self.query_prefix + self.content}')
            chat_completion = openai.ChatCompletion.create(model=self.choose_model, messages=[{"role": "user", "content": self.query_prefix + self.content}])
        except Exception as ex:
            reason = type(ex).__name__
            self.logger.error(f'访问模型{self.choose_model}时出错，错误原因::{ex}  错误类型:: {reason}')
            return
        
        end_time = datetime.now()
        duration = (end_time - begin_time).total_seconds()
        answer = chat_completion.choices[0].message.content
        self.answer = answer
        
        query_obj = {}
        
        request_id = str(uuid.uuid4())
        self.request_id = request_id

        self.logger.info(f'openai请求完成，此次请求id为 {request_id}， 请求时间 {duration}s')

        query_obj['request_id'] = request_id
        query_obj['model'] = self.choose_model
        query_obj['duration'] = (end_time - begin_time).total_seconds()
        query_obj['query'] = self.content
        query_obj['query_time'] = begin_time_str
        query_obj['answer'] = answer

        self.request_record.append(query_obj)

        dir_name = os.path.dirname(self.query_res_path)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not os.path.exists(self.query_res_path):
            with open(self.query_res_path, 'w'):
                pass

        with open(self.query_res_path, 'r+') as qop:
            qop.seek(0)
            full_content = qop.read()

            # 如果末尾有换行符，移除1个或多个换行符，统一加上一个换行符
            full_content = (full_content.rstrip('\n') + '\n') if full_content else ''

            for record in self.request_record:
                full_content += json.dumps(record, ensure_ascii=False) + '\n'
            qop.seek(0)
            qop.write(full_content)
            self.logger.info(f'request_record已写入路径:: {self.query_res_path}')

        self.logger.info(f'query已完成，模型的回答是:: {answer}')
        self.logger.info(f"{'-'*50}")

        # self.do_search()

    def send_search(self, query, request_id, index):
        conn = http.client.HTTPSConnection(self.search_domain)
        headers = {
            'Authorization': self.search_authorization,
            'content-type': self.search_content_type
            }

    
        payload = "{\"model\": \"" + self.search_model + "\",\"" + self.search_model.replace('-', '_') + "\": \"{\\\"q\\\":\\\"" + query.replace('"', "'") + "\\\"}\"}"
        payload_str = payload.encode('utf-8')
        conn.request("POST", self.search_sub_url, payload_str, headers)
        self.logger.info(f'发送request:: {query}')
        begin_time = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        
        time.sleep(2)
        res = conn.getresponse()
        data = res.read()

        res_str = data.decode(self.search_encode)

        res_obj = json.loads(res_str)
        
        self.logger.info(f'结果为:: {res_str}')

        search_obj = {
            'request_id': request_id,
            'request_index': index,
            'search_model': self.search_model,
            'search_time': begin_time,
            'search_query': query,
            'search_response': res_obj
        }
        self.search_record.append(search_obj)
        if len(self.search_record) % 3 == 0:
            self.save_search_res()
    
    def save_search_res(self):
        with open(self.search_res_path, 'r+') as srp:
            full_content = ''
            for record in self.search_record:
                full_content += json.dumps(record, ensure_ascii=False) + '\n'
            srp.seek(0)
            srp.write(full_content)


    '''
        执行do_search有三种方式：
        （1）单次调用，参数里必须有search_query参数
        （2）对某一大模型请求之后的结果调用，用request_id进行区分是哪次请求，request_index进行区分是请求中的哪个问句
        （3）从大模型的query开始执行，既不用传search_query，也不用传request_id，query后会自动赋值
    '''
    def do_search(self):
        # 如果yaml传了search.search_query参数，则直接根据这个query执行search
        if not os.path.exists(self.search_res_path):
            with open(self.search_res_path, 'w'):
                pass
        else:
            with open(self.search_res_path, 'r') as srp:
                for line in srp:
                    if line:
                        search_obj = json.loads(line)
                        self.search_record.append(search_obj)

        if self.search_query:
            self.send_search(self.search_query, '', -1)
            self.save_search_res()
            return
         
        # 如果yaml传了search.request_id参数，说明从之前大模型query的结果中查找，需要先根据这个id找到answer
        if hasattr(self, 'request_id') and (not hasattr(self, 'answer')):
            with open(self.query_res_path , 'r') as qop:
                for line in qop:
                    query_record = json.loads(line)
                    if query_record['request_id'] == self.request_id:
                        self.answer = query_record['answer']
                        break
        
        lines = self.answer.split('\n')
        re_num = r"\d+\."
        lines = [re.sub(re_num, '' , line, count=1).strip() for line in lines]
        self.question_list.extend(lines)
        
        for index, question in enumerate(self.question_list):
            self.send_search(question, self.request_id, index)

        self.save_search_res()
        
        self.logger.info(f'search已完成，模型的回答记录在:: {self.search_res_path}')
        self.logger.info(f"{'-'*50}")


    def selenium_get(self, url):
        try:
            self.driver.get(url)
        except Exception as ex:
            self.logger.info(f'selenium访问{url}出错!!')
            return ''
        start_time = datetime.now()
        timeout = timedelta(seconds = self.crawl_timeout)  # 等待10秒
        WebDriverWait(self.driver, self.crawl_timeout + 5, poll_frequency=1).until(
            lambda d: (datetime.now() - start_time) > timeout
        )
        content = self.simple_wash(url, self.driver.page_source)
        return content

    # 传入一个url和一个原始的html字符串内容，输出一个清洗后的页面提取结果
    def simple_wash(self, url, origin_text):
        try:
            soup = BeautifulSoup(origin_text, 'html.parser')
        except Exception:
            self.logger.info(f"{url} 封装成soup的过程出现异常！！")
            return False

        # 删除body中的全部footer和header元素  
        body_elements = soup.find_all('body')
        if len(body_elements) == 0:
            return False
            
        body_element = body_elements[0]
        footer_elements = body_element.find_all('footer')
        for footer_element in footer_elements:
            # 删除footer元素,footer后的所有元素也删除
            footer_element.extract()
            if footer_element.next_siblings:
                for sibling in footer_element.next_siblings:
                    sibling.extract()
                                
        header_elements = body_element.find_all('header')
        for header_element in header_elements:
            # 删除该元素
            header_element.extract()

        content = body_element.text
        line_pattern = r'(\n){3,}'
        whitespace_pattern = r'( ){3,}'
        table_pattern = r'(\t){2,}'

        content = re.sub(line_pattern, '\\n\\n', content)
        content = re.sub(whitespace_pattern, '  ', content)
        content = re.sub(table_pattern, '\\t', content)
        content = self.deal_with_sp(content)

        return content
    '''
        执行do_crawl有三种方式：
        （1）直接调用，参数里必须有crawl_urls参数
        （2）对某一大模型请求之后的结果调用，用request_id进行区分是哪次请求，crawl_search_index进行区分是请求返回结果中的哪个search，默认为0
        （3）从大模型的query开始执行，既不用传search_query，也不用传request_id，query、search后会自动赋值
    '''
    def do_crawl(self):
        # 当获取的页面文本长度小于100时认为页面没有加载完成
        # 这里需要先创建一个WebDriver
        s = Service(executable_path='/usr/local/bin/chromedriver')
        # 创建chrome配置
        options = Options()
        options.add_argument('--headless')  # 启用无头模式
        options.add_argument('--no-sandbox') # 启动无沙盒模式，否则会报错：chrome crashed
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36')
        # self.driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)
        # driver对象是selenium用于操作网页的对象，需要借助chromedriver实现
        self.driver = webdriver.Chrome(service=s, options=options)

        # 为简化配置，headers写死在代码里
        self.crawl_headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
        
        if not os.path.exists(self.crawl_res_path):
            with open(self.crawl_res_path, 'w'):
                pass
        else:
            with open(self.crawl_res_path, 'r') as crp:
                for line in crp:
                    if line:
                        crawl_obj = json.loads(line)
                        self.crawl_record.append(crawl_obj)

        self.check_crawl_dir()
        current_html_files = os.listdir(self.crawl_html_dir)
        if not len(current_html_files):
            self.html_count = 0
        else:
            current_html_files = [file for file in current_html_files if file.split('.')[0].isdigit()]
            current_html_files = sorted(current_html_files , key = lambda item : int(item.split('.')[0]))
            self.html_count = int(current_html_files[-1].split('.')[0])

        current_img_files = os.listdir(self.crawl_img_dir)
        if not len(current_img_files):
            self.img_count = 0
        else:
            current_img_files = [file for file in current_img_files if file.split('.')[0].isdigit()]
            current_img_files = sorted(current_img_files , key = lambda item : int(item.split('.')[0]))
            self.img_count = int(current_img_files[-1].split('.')[0])

        current_pdf_files = os.listdir(self.crawl_pdf_dir)
        if not len(current_pdf_files):
            self.pdf_count = 0
        else:
            current_pdf_files = [file for file in current_pdf_files if file.split('.')[0].isdigit()]
            current_pdf_files = sorted(current_pdf_files , key = lambda item : int(item.split('.')[0]))
            self.pdf_count = int(current_pdf_files[-1].split('.')[0])

        if self.crawl_urls:
            for url in self.crawl_urls:
                self.send_crawl(url, '', 0)
            self.save_crawl_res()
        else:
            # 找到request_id下的所有url
            if not self.search_record:
                if os.path.exists(self.search_res_path):
                    with open(self.search_res_path) as srp:
                        for line in srp:
                            self.search_record.append(json.loads(line))
                else:
                    self.logger.info(f'{self.search_res_path}不存在，没有提供正确的search_res_path!!!')
                    return
                
            for record in self.search_record:
                if self.request_id == record['request_id'] and self.crawl_search_index == record['request_index']:

                    res = record['search_response']
                    items = res['items']
                    for item in items:
                        self.crawl_urls.append(item['link'])
                        pagemap = item.get('pagemap')
                        cse_image = pagemap.get('cse_image') if pagemap else None
                        if cse_image:
                            cse_image_list = [item['src'] for item in pagemap.get('cse_image')]
                            self.crawl_urls.extend(cse_image_list)
                
                    for url in self.crawl_urls:
                        self.send_crawl(url, self.request_id, self.crawl_search_index)
                    self.save_crawl_res()
                    
                    if not self.crawl_single_batch:
                        self.crawl_search_index += 1
                        self.crawl_urls = []
                        self.html_count = 0
                        self.img_count = 0
                        self.pdf_count = 0
                else:
                    continue
        
        self.logger.info(f'crawl已完成，爬虫的记录存放在:: {self.crawl_res_path}')
        self.logger.info(f"{'-'*50}")


    # 创建存放html/img/pdf的三种不同文件夹
    # 如果有request_id，在data_dir/request_id/search_index/路径下创建
    # 如果没有，则直接在/data_dir路径下创建
    def check_crawl_dir(self):
        if not self.request_id:
            self.crawl_html_dir = os.path.join(self.crawl_data_dir, 'html')
            self.crawl_img_dir = os.path.join(self.crawl_data_dir, 'img')
            self.crawl_pdf_dir = os.path.join(self.crawl_data_dir, 'pdf')
        else:
            self.crawl_html_dir = os.path.join(self.crawl_data_dir, self.request_id, str(self.crawl_search_index), 'html')
            self.crawl_img_dir = os.path.join(self.crawl_data_dir, self.request_id, str(self.crawl_search_index), 'img')
            self.crawl_pdf_dir = os.path.join(self.crawl_data_dir, self.request_id, str(self.crawl_search_index), 'pdf')

        dir_list = [self.crawl_html_dir, self.crawl_img_dir, self.crawl_pdf_dir]
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
        
    def send_crawl(self, url, request_id, index):
        # 先查看是否建立对应的文件夹
        self.check_crawl_dir()
        # 使用requests进行请求
        # 根据返回的type类型进行不同处理
        self.logger.info(f'开启request_id为{request_id} index为 {index} 的url {url} 的爬取过程...................')
        try:
            # 设置一个timout，不然遇到网络问题会无限期挂起
            response = requests.get(url, verify=False, headers=self.crawl_headers, timeout=self.crawl_timeout)
            response.raise_for_status()  # 将捕获HTTP错误
        except RequestException as e:
            error_type = type(e).__name__
            self.logger.info(f"request_id为{request_id} index为{index}的请求 {url} 请求过程中发生了错误: {e} :::: 错误类型: {error_type}")
            return 
        
        status_code = str(response.status_code)
        self.logger.info(f"{url} 请求状态码为{status_code} ")
        if status_code[0] == '4' or status_code[0] == '5':
            return False
        
        content_type = response.headers.get('Content-Type').lower()
        self.logger.info(f"{url} 请求的content-type为{content_type}")
        
        if 'html' in content_type and 'html' in self.crawl_save_types:
            # html_dir = os.path.join(self.crawl_data_dir, 'html')
            # if not os.path.exists(html_dir):
            #     os.makedirs(html_dir)
            content = self.simple_wash(url, response.text)

            if content and len(content) < self.crawl_text_threshold:
                content = self.selenium_get(url)

            if not content:
                self.logger.info(f'{url} 数据获取失败!!!')
                return
            
            if len(content) < self.crawl_text_threshold:
                self.logger.info(f'{url} 获取txt长度过短，可能存在异常，请检查!!!')
                return

            # 要写入json文件的数据本身

            html_obj = {
                'request_id': self.request_id,
                'search_index': self.crawl_search_index,
                'from_url' : url,
                'text': content
            }
            
            current_file_name = f'{self.html_count}.json'
            full_path = os.path.join(self.crawl_html_dir, current_file_name)
            with open(full_path, 'w') as fp:
                fp.write(json.dumps(html_obj, ensure_ascii=False))
            self.logger.info(f'{current_file_name} 文件已写入路径 {self.crawl_html_dir} !!!!')
            # 爬取的记录
            crawl_record_obj = {
                # file_dir / file_name / file_type / file_size  
                'html_count': self.html_count,
                'request_id': self.request_id,
                'search_index': self.crawl_search_index,
                'from_url' : url,
                'crawl_time': datetime.now().strftime('%Y_%m_%d_%H:%M:%S'),
                'content_type': content_type,
                'file_dir': self.crawl_html_dir,
                'file_type': 'json',
                'file_size': os.path.getsize(full_path)
            }

            self.crawl_record.append(crawl_record_obj)
            self.html_count += 1
            if len(self.crawl_record) % 5 == 0:
                self.save_crawl_res()

        elif 'image' in content_type and 'img' in self.crawl_save_types:
            img_type = content_type.split('/')[-1]
            current_file_name = f'{self.img_count}.{img_type}'
            full_path = os.path.join(self.crawl_img_dir, current_file_name)
            with open(full_path, 'wb') as fp:
                fp.write(response.content)
            self.logger.info(f'{current_file_name} 图片已写入路径 {self.crawl_img_dir} 格式为{img_type}!!!!')
            crawl_record_obj = {
                'img_count': self.img_count,
                'request_id': self.request_id,
                'search_index': self.crawl_search_index,
                'from_url' : url,
                'crawl_time': datetime.now().strftime('%Y_%m_%d_%H:%M:%S'),
                'content_type': content_type,
                'file_dir': self.crawl_img_dir,
                'file_type': img_type,
                'file_size': os.path.getsize(full_path)
            }
            self.crawl_record.append(crawl_record_obj)
            self.img_count += 1
            if len(self.crawl_record) % 5 == 0:
                self.save_crawl_res()

        elif 'pdf' in content_type and 'pdf' in self.crawl_save_types:
            current_file_name = f'{self.pdf_count}.pdf'
            full_path = os.path.join(self.crawl_pdf_dir, current_file_name)
            with open(full_path, 'wb') as fp:
                fp.write(response.content)
            self.logger.info(f'{current_file_name} pdf文件已写入路径 {self.crawl_pdf_dir} !!!!')
            crawl_record_obj = {
                'img_count': self.pdf_count,
                'request_id': self.request_id,
                'search_index': self.crawl_search_index,
                'from_url' : url,
                'crawl_time': datetime.now().strftime('%Y_%m_%d_%H:%M:%S'),
                'content_type': content_type,
                'file_dir': self.crawl_pdf_dir,
                'file_type': 'pdf',
                'file_size': os.path.getsize(full_path)
            }
            self.crawl_record.append(crawl_record_obj)
            self.pdf_count += 1
            if len(self.crawl_record) % 5 == 0:
                self.save_crawl_res()
        else:
            return 
        
    def save_crawl_res(self):
        with open(self.crawl_res_path, 'r+') as crp:
            # srp.seek(0)
            # full_content = srp.read()

            # # 如果末尾有换行符，移除1个或多个换行符，统一加上一个换行符
            # full_content = (full_content.rstrip('\n') + '\n') if full_content else ''
            full_content = ''
            for record in self.crawl_record:
                full_content += json.dumps(record, ensure_ascii=False) + '\n'
            crp.seek(0)
            crp.write(full_content)
            self.logger.info(f'crawl_record保存到记录:{self.crawl_res_path}!!!')
            
# 处理\u开头的unicode编码，转为中文字符
    def deal_with_sp(self, json_str):
        pattern = r"\\u[0-9A-Fa-f]{4}"
        return re.sub(pattern, self.convert_special, json_str)
        
# re.sub中对于正则匹配字符串的转换函数
    def convert_special(self, match):
        text = match.group(0)
        try:
            converted = text.encode().decode('unicode_escape')
            return converted
        except Exception as ex:
            return ''

class ConfigLoadError(Exception):
    def __init__(self, config_key):
        if config_key.startswith('log'):
            super().__init__(f'日志无法建立，{config_key} 参数缺失！！')  # 初始化基类，设置错误信息
        else:
            super().__init__(f'配置加载失败，参数 {config_key} 缺失，请补充配置！！')  # 初始化基类，设置错误信息
        # self.code = code  # 自定义属性
        
        formatted_str_f = '\n'.join(f"{k}: {v}" for k, v in settings.settings['query']['must'].items())
        print('正确的必须配置的参数如下：')
        print(formatted_str_f)

class StepError(Exception):
    def __init__(self, step_list):
        warning_message = f'step_list {"->".join(step_list)} 执行步骤错误, 正确的步骤应该是: query -> search -> crawl'
        super().__init__(warning_message)
        print(warning_message)


if __name__ == '__main__':
    search = Search()
    search.load_config()
    if 'query' in search.step_list:
        search.do_query()
    if 'search' in search.step_list:
        search.do_search()
    if 'crawl' in search.step_list:
        search.do_crawl()
        