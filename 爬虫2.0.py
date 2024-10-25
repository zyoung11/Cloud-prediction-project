from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import os
import time
import requests
from datetime import datetime

# 设置Edge浏览器驱动路径
edge_driver_path = "C:/Download_center/APP/APPs/EdgeDriver/edgedriver_win32/msedgedriver.exe"
service = Service(edge_driver_path)

# 初始化浏览器
options = webdriver.EdgeOptions()
options.add_argument('--headless')  # 无头模式，不打开浏览器窗口
driver = webdriver.Edge(service=service, options=options)

# 访问目标网页
url = 'https://data.cma.cn/data/online.html?t=6'
driver.get(url)

# 等待页面加载完毕
time.sleep(5)  # 根据需要调整等待时间

# 创建保存图片的文件夹
if not os.path.exists('images'):
    os.makedirs('images')

# 爬取的起始和结束时间
start_time = datetime(2024, 9, 22, 18, 0)
end_time = datetime(2024, 8, 22, 0, 0)

while True:
    # 查找具有id='showimages'的图片
    image_element = driver.find_element(By.ID, 'showimages')
    
    # 获取图片URL
    img_url = image_element.get_attribute('src')
    
    if img_url:
        img_name = os.path.basename(img_url)
        img_datetime_str = img_name.split('_')[-1].replace('.png', '')
        img_datetime = datetime.strptime(img_datetime_str, '%Y%m%d%H%M%S')

        # 如果当前图片的时间早于结束时间，停止爬取
        if img_datetime < end_time:
            print("爬取结束")
            break

        # 下载并保存图片
        img_data = requests.get(img_url).content
        with open(f'images/{img_name}', 'wb') as handler:
            handler.write(img_data)
            print(f'{img_name} downloaded.')

        # 点击“上一张”按钮，获取上一小时的图片
        prev_button = driver.find_element(By.XPATH, "//span[text()='上一张']")
        ActionChains(driver).move_to_element(prev_button).click().perform()

        # 等待页面加载新图片
        time.sleep(00)
    else:
        print("未找到图片URL")
        break

# 关闭浏览器
driver.quit()
