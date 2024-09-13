# 版本迭代记录
| 时间 | 版本 | 说明 |
| --- | --- | --- |
| 2024.7.10 | V1.0 | 初版完成 |
| | | |

## 作者信息
如有问题，可联系作者：
周子豪 / 之江实验室
邮箱：zhouzih@zhejianglab.org
微信：wangchuan1434050333

# 环境依赖
1.安装依赖的python包
pip install -r requirements.txt

2.由于爬取使用ajax的网页时，需要借助selenium驱动服务器浏览器，因此需要在linux服务器上安装linux版
chrome浏览器及其驱动。

sudo apt-get update

wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

安装下载的.deb包：
使用dpkg命令安装Chrome。
sudo dpkg -i google-chrome-stable_current_amd64.deb

解决依赖问题（如果有的话）：
如果dpkg命令报告缺少依赖并失败，可以使用以下命令来修复依赖问题：
sudo apt-get -f install

清理（可选）：
安装完成后，删除下载的.deb文件以节省空间。
rm google-chrome-stable_current_amd64.deb

安装完成后，查看浏览器版本：
google-chrome --version

Linux安装ChromeDriver:

到网站上下载对应版本的ChromeDriver
https://googlechromelabs.github.io/chrome-for-testing/
unzip chromedriver_linux64.zip
sudo cp chromedriver /usr/local/bin/chromedriver
sudo chmod +x /usr/local/bin/chromedriver
chromedriver --version

# 示例
这里选用的大模型为gpt-4，对模型的输入为：

Now I am the game organizer of StarCraft 2 and I need to test some players' understanding of this game. Please help me write some questions?

注：（1）这里需要加上对大模型输出回答格式的要求作为prompt，以便于对大模型的回复进行解析。（2）如果需要限制大模型回复的数量，不至于过多或过少，可以在配置文件中设置希望大模型给到的答案数，默认为20。

因此完整的输入为：

```plain
Please answer my question strictly according to the following template :

 1.answer1 
 2.answer2 
...
 I need 20 answers of the question. The question is: 
I am the game organizer of StarCraft 2 and I need to test some players' understanding of this game. Please help me write some questions?
```



大模型的输出为：

```yaml
1. What is the primary resource used to build units in StarCraft 2?
2. Which race in StarCraft 2 has the ability to use the Warp-in mechanic?
3. What unit is produced from the Starport with a Tech Lab attached in StarCraft 2?
4. What is the name of the Zerg worker unit in StarCraft 2?
5. How many minerals does a Marine cost in StarCraft 2?
6. Which Protoss unit is known for its cloaking ability in StarCraft 2?
7. What upgrade increases the movement speed of Zerglings in StarCraft 2?
8. How many Vespene Geysers are typically found at a standard expansion location in StarCraft 2?
9. What is the function of the Terran's Orbital Command in StarCraft 2?
10. What is the build time of a Protoss Zealot in StarCraft 2?
11. What ability does the Terran Raven's Auto-Turret provide in StarCraft 2?
12. Which Zerg unit can transform into Banelings in StarCraft 2?
13. What is the name of the Protoss capital ship in StarCraft 2?
14. How do you achieve a victory in a game of StarCraft 2?
15. What is 'creep' and how does it benefit Zerg in StarCraft 2?
16. Name the three races playable in StarCraft 2.
17. Which unit can be produced at the Protoss Robotics Facility in StarCraft 2?
18. What is the function of the Terran SCV in StarCraft 2?
19. How does the Terran Siege Tank change when it enters Siege Mode in StarCraft 2?
20. What Zerg building is required to spawn Mutalisks in StarCraft 2?  
```



把大模型输出的第一个question作为query，查询了google-search后，得到一组url：

[https://tictactactics.wordpress.com/2015/07/26/the-beginners-guide-to-starcraft-2-part-iii-choosing-a-race/"](https://tictactactics.wordpress.com/2015/07/26/the-beginners-guide-to-starcraft-2-part-iii-choosing-a-race/",)

https://tl.net/forum/sc2-maps/361038-how-to-design-a-new-race-for-starcraft-2

.....

# 工具介绍
基于上述逻辑：给大模型输入 -> 大模型输出作为query输入到google-search -> 爬取google-search输出的url

## 功能点
### 主要功能点
#### 大模型查询
指定模型种类，指定输入到大模型的input，查询对应的结果。

模型种类由query.choose_model参数指定

输入到大模型的input由query.content指定


#### gooogle-search
指定query，调用google-search api获取需要的结果。

可以添加search.search_query参数指定一次search，也可以通过传递request_id（大模型查询任务的唯一标志）和search.query_res_path参数（指定大模型查询的记录）来根据之前大模型查询的结果，进行gooogle-search查询。

#### 数据爬取
指定urls进行数据爬取。

可以添加crawl.crawl_urls参数指定url集进行爬取，也可以通过传递request_id（大模型查询任务的唯一标志）和search.search_res_path参数（指定gooogle-search的记录）来根据之前gooogle-search查询的结果，进行数据爬取。

以上三者即可以串联实现，也可以单独调用。

决定是否串联实现的配置：

配置文件中的step_list


### 其他功能点
#### 集中配置
所有需要的配置在一个yaml文件中定义。

默认的yaml文件为：one_click_crawler.yaml

可以自定义配置文件路径，在调用时用--config参数指明即可。

#### 处理记录保存
大模型、google-search模型调用费用都比较高，需要对每次模型的调用结果进行保存。

query.query_res_path指定大模型调用的处理结果存储路径

search.search_res_path指定google-search的查询结果

crawl.crawl_res_path指定爬虫的处理结果

以上参数指定的文件都必须是jsonl格式，可以不存在，如果不存在会在调用时进行创建

#### 日志记录
每次工具调用的日志记录，方便对结果进行回溯。

#### 过滤爬虫文件种类
可以指定爬取的格式，目前支持三种：

html：html网页，会默认提取所有文字并进行简单清洗（复杂的数据清洗需要调用数据处理工具）

img：google-search返回结果中网页上出现的图片

pdf：google-search返回结果中的pdf

可以在参数crawl.crawl_save_types中指定需要获取哪些数据类型。


## 配置文件
可以参考项目路径下的setting.py文件，文件中对哪些配置项是必须的、哪些配置项是可选的，以及对应的配置规则都做了说明。

```python
# 配置规则
# query和search和crawl三者互斥，配置文件中有query，就从query开始
# 若没有query，有search，就从search开始
# 若没有query和search，但是有crawl，就从crawl开始

# 可选择的大模型列表如下：
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

settings = {
    'log_dir': 'log日志的存放路径',
    'logger_name':'日志的名字',
    'query':{
        'must':{
            'choose_model': '指定的model',
            'query_res_path': 'query结果存放的路径'
        },
        'optional': {
            'api_base': '请求的api路径', 
            'api_key': '请求需要的key',
            'content': '请求需要的query'
        }
    },
    'search':{
        'must': {
            'search_res_path': 'search结果存放的路径'
        },
        'optional': {
            'search_domain' : 'search请求发送的domain',
            'search_query' : 'search发送的query',
            'search_model' : 'search选择的model',
            'search_authorization': 'search的key',
            'search_content_type': 'search的contenttype，默认为application/json',
            'search_sub_url': 'search的子路径，默认为/v1/completions',
            'search_encode': 'search结果的编码方式，默认为utf-8',
            'request_id': 'request的id，基于之前大模型查询的结果时需要',
            'query_res_path': 'query结果存放的路径，如果只提供request_id，一定需要带上query_res_path'
        }
    },
    'crawl':{
        'must': {
            'crawl_data_dir': '爬虫结果存放的路径',
            'crawl_res_path': '爬虫记录文件路径'
        },
        'optional': {
            'crawl_urls': '需要爬取的url，列表类型，如果传入的话，后面的request_id等失效',
            'request_id': 'request的id，基于之前大模型查询的结果时需要',
            'crawl_search_index': '这个request_id下对应的search的index，对应第index个query对应的google_search结果，默认为0',
            'crawl_save_types': '需要保存的种类，如需要爬取html网页内容、图片、pdf等，列表类型，默认为[html, img, pdf]',
            'search_res_path': '如果指明了request_id，必须指定search_res_path，否则无法加载到记录',
            'crawl_timeout': '等待request的最长时间，单位s，默认为60',
            'crawl_text_threshold': '从html页面上获取text的最低门限，超过这个门限则进入selenium的代码，执行等待',
            'crawl_single_batch': '是否只爬取一个批次，默认为False'
        }
    }
}
```

# 使用教程
## 创建配置文件
默认的配置文件为项目目录下configs/one_click_crawler.yaml

默认配置如下：

```yaml
step_list:
  - query
  - search
  - crawl

log_dir: /home/zhouzihao/one_click_crawler/logs
logger_name: query
 
query:
  content: "I am the game organizer of StarCraft 2 and I need to test some players' understanding of this game. Please help me write some questions?"
  choose_model: gpt-4
  query_res_path: /home/zhouzihao/data/crawler/gpt4/query_res.jsonl

search:
  query_res_path: /home/zhouzihao/data/crawler/gpt4/query_res.jsonl
  search_res_path: /home/zhouzihao/data/crawler/gpt4/search_res.jsonl

crawl:
  crawl_data_dir: /home/zhouzihao/data/crawler/gpt4/data
  crawl_res_path: /home/zhouzihao/data/crawler/gpt4/crawl_res.jsonl
```

需要注意以下核心参数：

| 序号 | 参数名 | 是否必须 | 意义 | 默认值 |
| --- | --- | --- | --- | --- |
| 1 | step_list | 是 | 需要进行的操作步骤 | 无（必须在配置文件中指定） |
| 2 | log_dir | 是 | 日志存放目录 | 无（必须在配置文件中指定） |
| 3 | logger_name | 是 | logger模块的名字，用于标志此次运行 | 无（必须在配置文件中指定） |
| 4 | query_res_path | 如果需要进行大模型查询，则必须 | 大模型查询结果保存路径 | 无（必须在配置文件中指定） |
| 5 | search_res_path | 如果需要进行google-search查询，则必须 | google-search查询结果保存路径 | 无（必须在配置文件中指定） |
| 6 |   crawl_data_dir | 如果需要进行爬虫，则必须 | 爬取的数据保存路径 | 无（必须在配置文件中指定） |
| 7 |  crawl_res_path | 如果需要进行爬虫，则必须 | 爬取记录 | 无（必须在配置文件中指定） |
| 8 | query.content | 如果需要进行大模型查询，则必须 | 输入给大模型的query问句 | 无（必须在配置文件中指定） |


## 运行代码

运行程序：

python one_click_crawler.py

可使用--config参数指定配置文件路径，若不指定则使用默认的配置文件，默认配置文件为：configs/one_click_crawler.yaml



