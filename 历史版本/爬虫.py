from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
import os
import time
import requests

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

# 查找具有id='showimages'的图片
image_element = driver.find_element(By.ID, 'showimages')

# 获取图片URL
img_url = image_element.get_attribute('src')

# 下载图片
if img_url:
    img_name = os.path.basename(img_url)

    # 创建保存图片的文件夹
    if not os.path.exists('images'):
        os.makedirs('images')

    # 下载并保存图片
    img_data = requests.get(img_url).content
    with open(f'images/{img_name}', 'wb') as handler:
        handler.write(img_data)
        print(f'{img_name} downloaded.')

# 关闭浏览器
driver.quit()
