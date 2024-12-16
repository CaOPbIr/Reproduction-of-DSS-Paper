import time
import random
import io
import sys
import re
from loguru import logger
from DrissionPage import Chromium, ChromiumOptions
from bs4 import BeautifulSoup
from pymongo import MongoClient
from multiprocessing import Process

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')  

def get_data(tab):
    eles = tab.eles('.comment-item')
    data_ls = []
    for ele in eles:
        cleaned_data = data_clean(ele) 
        data_ls.append(cleaned_data)
    return data_ls

def turn_page(tab):
    tab.set.scroll.smooth(on_off=False)
    tab.actions.scroll(random.randint(2800,3200)).move_to('.ui-pager-next').click()
    time.sleep(random.uniform(1, 2))

def data_clean(element):
    data = element.html
    soup = BeautifulSoup(data, 'html.parser')
    comment = soup.find('p', class_='comment-con')
    rating = soup.find('div', class_='comment-star')['class'][1]
    is_plus = True if soup.find('div', class_='user-level').find('a') else False
    helpful_vote = soup.find('a', class_='J-nice').text
    release_time = soup.find('span', string=re.compile('\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])'))
    childs = soup.find('div', class_='comment-column J-comment-column').find_all('div')
    has_image = True if childs[1].get('class')==['pic-list', 'J-pic-list'] else False
    image_num = len(childs[1].find_all(class_='J-thumb-img'))
    has_video = True if childs[3].get('class')==['J-video-view-wrap', 'clearfix'] else False
    has_replies = True if int(soup.find('i', class_='sprite-comment').parent.text) != 0 else False
    if has_replies:
        replies_page = element.ele('.sprite-comment').click.for_new_tab()
        replies_page.wait.load_start()
        replies_ls = replies_page.s_eles('.tt')
        replies = [reply.text.split(': ')[1] for reply in replies_ls]
        replies_page.close()
    return {'Comment':comment.text, 'Rating':rating, 'Is_plus':is_plus, 'Helpful_votes':helpful_vote, 'Release_time':release_time.text, 'Has_image':has_image, 'Image_num':image_num, 'Has_video':has_video, 'Replies':[] if not has_replies else replies}

def get_product_ids(name):
    co = ChromiumOptions().set_local_port(9000)
    browser = Chromium(co)
    detail_page = browser.new_tab(f'https://search.jd.com/Search?keyword={name}&enc=utf-8&wq={name}')
    detail_page.ele('@@class=fs-tit@@text()=销量').click()
    time.sleep(2)
    detail_page.scroll(20000)
    time.sleep(2)
    products = detail_page.eles('.gl-item')
    return [product.attr('data-sku') for product in products]


def scrape_product(port, name, product_id):
    co = ChromiumOptions().set_local_port(port).add_extension(r"D:\AppsLite\SwitchyOmega_Chromium")
    browser = Chromium(co)
    
    client = MongoClient('localhost', 27017)
    db = client[name]
    collection = db[product_id]
    
    main_tab = browser.new_tab(f'https://item.jd.com/{int(product_id)}.html#comment')

    i, j = 0, 0
    max_tries = 6
    while i < 10:
        try:
            data = get_data(main_tab)
            collection.insert_many(data)
            turn_page(main_tab)
            logger.success(f'商品{product_id}的第{i+1}页数据已保存')
            i += 1
            j = 0
        except Exception as e:
            logger.error(f'出现错误: {str(e)}，正在尝试刷新页面')
            main_tab.refresh()
            time.sleep(5)
            j += 1
            if j > max_tries:
                break
            continue


if __name__ == '__main__':
    names = ['手机','笔记本电脑','数码相机','唇膏','运动鞋', '电饭煲']
    for name in names:
        product_ids = get_product_ids(name)
        ports = [9007, 9002, 9003, 9004, 9005, 9006]
        processes = []
        logger.info(f'开始抓取 {name} 数据, 共 {len(product_ids)} 个商品')

        for turn in range(10):
            ids = product_ids[turn*6:(turn+1)*6]
            for port, product_id in zip(ports, ids):
                p = Process(target=scrape_product, args=(port, name, product_id))
                processes.append(p)
                p.start()
        
            for p in processes:
                p.join()
        
            logger.success(f'{name}的第{turn+1}轮数据抓取完成')