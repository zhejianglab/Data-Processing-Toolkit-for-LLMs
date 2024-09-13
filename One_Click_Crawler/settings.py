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
            'content': '请求需要的query',
            'query_count': '需要大模型回答的answer数量，默认20'
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
            'crawl_text_threshold': '从html页面上获取text的最低门限，低于这个门限则进入selenium的代码，执行等待',
            'crawl_single_batch': '是否只爬取一个批次，默认为False'
        }
    }
}

