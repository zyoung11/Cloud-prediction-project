document.addEventListener('DOMContentLoaded', function() {
    const uploadsFolder = 'imatest1'; // 替换为实际路径
    const imagesPerPage = 11;
    let currentPage = 0;
    let currentImageIndex = 0;
    let images = [];
    let intervalId = null; // 初始化为 null;

    // 模拟获取图片列表
    function fetchImages() {
        const imagePaths = [
            '20240702080000.png',
            '20240702081000.png',
            '20240702082000.png',
            '20240702083000.png',
            '20240702084000.png',
            '20240702085000.png',
            '20240702090000.png',
            '20240702091000.png',
            '20240702092000.png',
            '20240702093000.png',
            '20240702094000.png',
            '1h_colored_prediction_1.png',
            '20240702100000.png',
            '20240702101000.png',
            '20240702102000.png',
            '20240702103000.png',
            '20240702104000.png',
            '20240702105000.png',
            '20240702110000.png',
            '20240702111000.png',
            '20240702112000.png',
            '20240702113000.png',
            '20240702114000.png',
            '20240702115000.png',
            '20240702120000.png',
            '20240702121000.png',
            '20240702122000.png',
            '20240702123000.png',
            '20240702124000.png',
            '20240702125000.png',
            '20240702130000.png',
            '20240702131000.png',
            '20240702132000.png',
            '20240702133000.png',
            '20240702134000.png',
            '20240702135000.png',
            '20240702140000.png',
            '20240702141000.png',
            '20240702142000.png',
            '20240702143000.png',
            '20240702144000.png',
            '20240702145000.png',
            '20240702150000.png',
            '20240702151000.png',
            '20240702152000.png',
            '20240702153000.png',
            '20240702154000.png',
            '20240702155000.png',
            '20240706050000.png',
            '20240706051000.png',
            '20240706052000.png',
            '20240706053000.png',
            '20240706054000.png',
            '1h_colored_prediction_0.png'
        ];
        const fullPaths = imagePaths.map(img => `${uploadsFolder}/${img}`);
        console.log('Loaded images:', fullPaths);
        return fullPaths;
    }

    // 初始化图片列表
    function initializeImages() {
        images = fetchImages();
        if (images.length === 0) {
            console.error('No images loaded.');
            return;
        }
        createImageList();
        loadPage(currentPage);
        updatePagination(currentPage);
    }

    // 创建图片列表
    function createImageList() {
        const imageList = document.getElementById('image-list1');
        imageList.innerHTML = '';
        images.forEach((img, index) => {
            const li = document.createElement('li');
            li.dataset.index = index;
            li.textContent = img.split('/').pop();
            li.addEventListener('click', function() {
                const index = this.dataset.index;
                showImage(index - (currentPage * imagesPerPage));
            });
            imageList.appendChild(li);
        });
    }

    // 加载指定页数的图片
    function loadPage(page) {
        const start = page * imagesPerPage;
        const end = start + imagesPerPage;
        const pageImages = images.slice(start, end);

        const slider = document.getElementById('image-slider1');
        slider.innerHTML = '';
        pageImages.forEach((img, index) => {
            const imgElement = document.createElement('img');
            imgElement.src = img;
            imgElement.alt = 'image';
            imgElement.style.display = 'none'; // 初始状态下隐藏所有图片
            imgElement.addEventListener('click', function() {
                showImage(index);
            });
            slider.appendChild(imgElement);
        });

        // 更新图片名称列表
        updateNameList(page);

        // 显示第一页的第一张图片
        showImage(0);
    }

    // 更新图片名称列表
    function updateNameList(page) {
        const start = page * imagesPerPage;
        const end = start + imagesPerPage;
        const pageNames = images.slice(start, end).map(img => img.split('/').pop());

        const nameList = document.getElementById('image-list1');
        nameList.innerHTML = '';
        pageNames.forEach((name, index) => {
            const li = document.createElement('li');
            li.textContent = name;
            li.addEventListener('click', function() {
                const pageIndex = currentPage * imagesPerPage + index;
                showImage(pageIndex);
            });
            nameList.appendChild(li);
        });
    }

    // 显示指定索引的图片
    function showImage(index) {
        const images = document.querySelectorAll('#image-slider1 img');
        images.forEach(img => img.style.display = 'none'); // 隐藏所有图片
        if (index >= 0 && index < images.length) {
            images[index].style.display = 'block'; // 显示指定索引的图片
            currentImageIndex = index;

            // 更新图片列表中的高亮项
            const imageListItems = document.querySelectorAll('#image-list1 li');
            imageListItems.forEach(item => item.classList.remove('highlight'));
            imageListItems[currentPage * imagesPerPage + index].classList.add('highlight');

            // 更新图片名称列表中的高亮项
            const nameListItems = document.querySelectorAll('#image-list1 li');
            nameListItems.forEach(item => item.classList.remove('highlight'));
            nameListItems[index].classList.add('highlight');
        } else if (index === images.length) { // 如果索引超出当前页面范围
            nextPage(); // 跳转到下一页
            showImage(0); // 并显示下一页的第一张图片
        }
    }

    // 更新分页按钮
    function updatePagination(page) {
        const pagination = document.getElementById('pagination1');
        if (!pagination) {
            console.error('Pagination element not found');
            return;
        }
        console.log('Updating pagination for page:', page);
        pagination.innerHTML = '';
    
        const totalPages = Math.ceil(images.length / imagesPerPage);
        console.log('Total pages:', totalPages);
        for (let i = 0; i < totalPages; i++) {
            const btn = document.createElement('button');
            btn.textContent = i + 1;
            btn.dataset.page = i;
            if (i === page) {
                btn.classList.add('active');
            }
            btn.addEventListener('click', function() {
                const page = this.dataset.page;
                currentPage = parseInt(page);
                loadPage(currentPage);
                updatePagination(currentPage);
            });
            pagination.appendChild(btn);
        }
    }

    // 上一页
    function prevPage() {
        const totalPages = Math.ceil(images.length / imagesPerPage);
        currentPage = (currentPage - 1 + totalPages) % totalPages;
        loadPage(currentPage);
        updatePagination(currentPage);
    }

    // 下一页
    function nextPage() {
        const totalPages = Math.ceil(images.length / imagesPerPage);
        currentPage = (currentPage + 1) % totalPages;
        loadPage(currentPage);
        updatePagination(currentPage);
    }

    // 初始化页面
    initializeImages();

    // 绑定事件
    document.getElementById('start-play1').addEventListener('click', startPlay);
    document.getElementById('stop-play1').addEventListener('click', stopPlay);
    document.getElementById('prev-page').addEventListener('click', prevPage);
    document.getElementById('next-page').addEventListener('click', nextPage);

    // 开始自动播放
function startPlay() {
    intervalId = setInterval(() => {
        const nextIndex = currentImageIndex + 1;
        if (nextIndex >= imagesPerPage) {
            nextPage();
            showImage(0);
        } else {
            showImage(nextIndex);
        }
    }, 1000); // 每3秒切换一次图片
}

// 停止自动播放
function stopPlay() {
    clearInterval(intervalId);
}
});