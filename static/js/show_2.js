document.addEventListener('DOMContentLoaded', function() {
    const uploadsFolders = ['imatest1', 'imatest2', 'imatest3']; // 各个区块对应的文件夹
    const imagesPerPage = 11; // 每页显示的图片数量
    let currentPage = [0, 0, 0]; // 每个区块的当前页数
    let currentImageIndex = [0, 0, 0]; // 每个区块的当前图片索引
    let images = [[], [], []]; // 每个区块的图片列表
    let intervalIds = [null, null, null]; // 每个区块的自动播放间隔ID

    const blocks = [document.getElementById('block1'),
        document.getElementById('block2'), 
        document.getElementById('block3')];

// 从服务器获取图片列表
function fetchImages(folder, callback) {
fetch(`/api/imatest/${folder}`)
.then(response => {
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return response.json();
})
.then(data => {
    const fullPaths = data.map(img => `${folder}/${img}`);
    console.log(`Loaded images from ${folder}:`, fullPaths);
    callback(fullPaths);
})
.catch(error => {
    console.error('Error fetching images:', error);
    callback([]);
});
}

// 初始化图片列表
function initializeImages(blockIndex) {
const folder = uploadsFolders[blockIndex];
fetchImages(folder, (fetchedImages) => {
images[blockIndex] = fetchedImages;
if (images[blockIndex].length === 0) {
    console.error(`No images loaded for block ${blockIndex}.`);
    return;
}
console.log(`Block ${blockIndex} has ${images[blockIndex].length} images.`);
createImageList(blockIndex);
loadPage(currentPage[blockIndex], blockIndex);
updatePagination(currentPage[blockIndex], blockIndex);
});
}

// 创建图片列表
function createImageList(blockIndex) {
const imageList = document.getElementById(`image-list${blockIndex + 1}`);
imageList.innerHTML = '';
images[blockIndex].forEach((img, index) => {
const li = document.createElement('li');
li.dataset.index = index;
li.textContent = getFileNameWithoutExtension(img.split('/').pop());
li.addEventListener('click', function() {
    const index = this.dataset.index;
    showImage(index, blockIndex);
});
imageList.appendChild(li);
});
}

// 加载指定页数的图片
function loadPage(page, blockIndex) {
const start = page * imagesPerPage;
const end = start + imagesPerPage;
const pageImages = images[blockIndex].slice(start, end);

const slider = document.getElementById(`image-slider${blockIndex + 1}`);
slider.innerHTML = '';
pageImages.forEach((img, index) => {
const imgElement = document.createElement('img');
imgElement.src = `/imatest/${img}`; // 确保路径正确
imgElement.alt = 'image';
imgElement.style.display = 'none'; // 初始状态下隐藏所有图片
imgElement.addEventListener('click', function() {
    showImage(index, blockIndex);
});
slider.appendChild(imgElement);
});

// 更新图片名称列表
updateNameList(page, blockIndex);

// 显示第一页的第一张图片
showImage(0, blockIndex);
}

// 更新图片名称列表
function updateNameList(page, blockIndex) {
const start = page * imagesPerPage;
const end = start + imagesPerPage;
const pageNames = images[blockIndex].slice(start, end).map(img => getFileNameWithoutExtension(img.split('/').pop()));

const nameList = document.getElementById(`image-list${blockIndex + 1}`);
nameList.innerHTML = '';
pageNames.forEach((name, index) => {
const li = document.createElement('li');
li.textContent = name;
li.addEventListener('click', function() {
    const pageIndex = start + index;  // 修复此处，确保索引正确
    showImage(pageIndex % imagesPerPage, blockIndex);  // 修改显示逻辑，确保在当前页显示对应图片
});
nameList.appendChild(li);
});
}

// 获取文件名并去除扩展名
function getFileNameWithoutExtension(fileName) {
return fileName.split('.').slice(0, -1).join('.');
}

// 显示指定索引的图片
function showImage(index, blockIndex) {
const images = document.querySelectorAll(`#image-slider${blockIndex + 1} img`);
console.log(`Total images in block ${blockIndex}:`, images.length); // 调试信息
images.forEach(img => {
img.style.display = 'none'; // 隐藏所有图片
img.classList.remove('active'); // 移除 active 类
});
if (index >= 0 && index < images.length) {
requestAnimationFrame(() => {
    images[index].style.display = 'block'; // 显示指定索引的图片
    images[index].classList.add('active'); // 添加 active 类
    currentImageIndex[blockIndex] = index;
    console.log(`Showing image at index ${index} in block ${blockIndex}`); // 调试信息
});

// 更新图片名称列表中的高亮项
const imageListItems = document.querySelectorAll(`#image-list${blockIndex + 1} li`);
if (imageListItems.length > 0) {
    imageListItems.forEach(item => item.classList.remove('highlight'));
    const highlightIndex = currentPage[blockIndex] * imagesPerPage + index;
    if (highlightIndex < imageListItems.length) {
        imageListItems[highlightIndex].classList.add('highlight');
    }
}
} else if (index === images.length) { // 如果索引超出当前页面范围
nextPage(blockIndex); // 跳转到下一页
showImage(0, blockIndex); // 并显示下一页的第一张图片
}
}

// 更新分页按钮
function updatePagination(page, blockIndex) {
const pagination = document.getElementById(`pagination${blockIndex + 1}`);
if (!pagination) {
console.error('Pagination element not found');
return;
}
console.log(`Updating pagination for page: ${page}, block: ${blockIndex}`);
pagination.innerHTML = '';

const totalPages = Math.ceil(images[blockIndex].length / imagesPerPage);
console.log(`Total pages for block ${blockIndex}: ${totalPages}`);
for (let i = 0; i < totalPages; i++) {
const btn = document.createElement('button');
btn.textContent = i + 1;
btn.dataset.page = i;
if (i === page) {
    btn.classList.add('active');
}
btn.addEventListener('click', function() {
    const page = this.dataset.page;
    currentPage[blockIndex] = parseInt(page);
    loadPage(currentPage[blockIndex], blockIndex);
    updatePagination(currentPage[blockIndex], blockIndex);
});
pagination.appendChild(btn);
}
}

// 上一页
function prevPage(blockIndex) {
const totalPages = Math.ceil(images[blockIndex].length / imagesPerPage);
currentPage[blockIndex] = (currentPage[blockIndex] - 1 + totalPages) % totalPages;
loadPage(currentPage[blockIndex], blockIndex);
updatePagination(currentPage[blockIndex], blockIndex);
}

// 下一页
function nextPage(blockIndex) {
const totalPages = Math.ceil(images[blockIndex].length / imagesPerPage);
currentPage[blockIndex] = (currentPage[blockIndex] + 1) % totalPages;
loadPage(currentPage[blockIndex], blockIndex);
updatePagination(currentPage[blockIndex], blockIndex);
}

// 控制区块的显示和隐藏
function showBlock(blockIndex) {
    blocks.forEach((block, index) => {
        block.classList.toggle('active', index === blockIndex);
        console.log(`Block ${index} is ${index === blockIndex ? 'active' : 'inactive'}`);
    });
    currentPage[blockIndex] = 0;
    currentImageIndex[blockIndex] = 0;
    initializeImages(blockIndex);
}

// 初始化页面
showBlock(0);

// 绑定事件
function bindEvents(blockIndex) {
document.getElementById(`show-block${blockIndex + 1}`).addEventListener('click', () => showBlock(blockIndex));
document.getElementById(`start-play${blockIndex + 1}`).addEventListener('click', () => startPlay(blockIndex));
document.getElementById(`stop-play${blockIndex + 1}`).addEventListener('click', () => stopPlay(blockIndex));
document.getElementById(`prev-page${blockIndex + 1}`).addEventListener('click', () => prevPage(blockIndex));
document.getElementById(`next-page${blockIndex + 1}`).addEventListener('click', () => nextPage(blockIndex));
}

bindEvents(0);
bindEvents(1);
bindEvents(2);

// 开始自动播放
function startPlay(blockIndex) {
intervalIds[blockIndex] = setInterval(() => {
const nextIndex = currentImageIndex[blockIndex] + 1;
if (nextIndex >= imagesPerPage) {
    nextPage(blockIndex);
    showImage(0, blockIndex);
} else {
    showImage(nextIndex, blockIndex);
}
}, 500); // 每3秒切换一次图片
}

// 停止自动播放
function stopPlay(blockIndex) {
clearInterval(intervalIds[blockIndex]);
intervalIds[blockIndex] = null; // 清除间隔ID
}

});