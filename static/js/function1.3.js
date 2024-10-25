document.addEventListener('DOMContentLoaded', function() {    
    const fileInput = document.getElementById('file-input');    
    const uploadBtn = document.getElementById('upload-btn');    
    const processBtn = document.getElementById('process-btn');      
    const resultsTable = document.getElementById('results-table');    
    const tbody = resultsTable.querySelector('tbody');    
    const prevBtn = document.getElementById('prev-btn');    
    const nextBtn = document.getElementById('next-btn');    
    const pageInfo1 = document.getElementById('page-info1');    
    const pageInfo2 = document.getElementById('page-info2');    
    let uploadedFiles = [];    
    let processedResults = [];    
    let currentPage = 0;   
    let currentAnimationPage = 0; 
    const itemsPerPage = 25;    
    let filenames = []; // 存储文件名

    // 绑定文件选择和上传按钮的点击事件
    uploadBtn.addEventListener('click', function() {
        fileInput.click(); // 触发文件选择对话框
    });

    fileInput.addEventListener('change', function() {    
        if (this.files.length > 0) {
            uploadFiles(this.files);
        }
    });

    function uploadFiles(files) {
        const formData = new FormData();    
        for (let file of files) {    
            formData.append('file', file);    
        }    
        fetch('/upload', {    
            method: 'POST',    
            body: formData    
        })    
        .then(response => response.json())    
        .then(data => {    
            if (data.error) {    
                logMessage('Error: ' + data.error);    
            } else {    
                logMessage('Files uploaded successfully. File count: ' + data.file_count);    
                uploadedFiles = Array.from(files).map(file => file.name);    
                processBtn.disabled = false;    
                fetchFilenames(); // 在文件上传成功后调用此函数
            }    
        })    
        .catch(error => {
            logMessage('Error uploading file: ' + error);
            console.error('Upload error:', error);
        });
        
    }

    processBtn.addEventListener('click', function() {    
        fetch('/process', {    
            method: 'POST',    
            headers: {    
                'Content-Type': 'application/json'    
            },    
            body: JSON.stringify({files: uploadedFiles})    
        })    
        .then(response => response.json())    
        .then(data => {    
            if (data.error) {    
                logMessage('Error: ' + data.error);    
            } else {    
                logMessage('Processing complete');    
                console.log('Data received from server:', data);  // 添加调试信息
                if (Array.isArray(data.results)) {
                    processedResults = data.results;
                    logMessage('Processed Results Length: ' + processedResults.length);
                } else {
                    logMessage('Error: data.results is not an array');
                    processedResults = [];  // 确保 processedResults 是一个数组
                }
                displayPage(currentPage);    
            }    
        })    
        .catch(error => logMessage('Error processing files: ' + error));    
    });

    function logMessage(message) {    
        const logDiv = document.getElementById('log'); // 获取日志区域
        const p = document.createElement('p');    
        p.textContent = message;    
        logDiv.appendChild(p);    
        logDiv.scrollTop = logDiv.scrollHeight;    
    }
    
    function displayPage(page) {
        const itemsPerPage = 25; // 每页25张图片
        const startIndex = page * itemsPerPage;
    
        // 1. 初始化表格行
        for (let i = 0; i < 5; i++) {
            const row = tbody.rows[i];
            for (let j = 0; j < 6; j++) {
                const cell = row.cells[j];
                cell.innerHTML = '';  // 清空之前的图片
                cell.style.display = 'none';  // 默认隐藏单元格
            }
        }
    
        // 2. 按时间分组并根据文件名尾部数字排序
        const timeGroups = {
            '10min': [],
            '30min': [],
            '1h': [],
            '2h': [],
            '3h': []
        };
    
        // 将文件按时间段分组
        processedResults.forEach(result => {
            const match = result.colored_output_path.match(/(10min|30min|1h|2h|3h)_模型\.pth_colored_prediction_(\d+)/);
            if (match) {
                const time = match[1];
                const order = parseInt(match[2], 10);
                if (timeGroups[time]) {
                    timeGroups[time].push({ ...result, order });
                }
            }
        });
    
        // 每个时间段的图片按 order 排序
        Object.keys(timeGroups).forEach(time => {
            timeGroups[time].sort((a, b) => a.order - b.order);
        });
    
        // 3. 按时间依次填充表格，每行填一组数据，每页25张图片（5行，每行5组）
        for (let i = 0; i < 5; i++) { // 5行
            const row = tbody.rows[i];
            const cell0 = row.cells[0];
            cell0.style.display = 'table-cell';
    
            // 填充每行的文件名（使用第一个时间段的文件名作为组名）
            const groupNameIndex = startIndex + i;
            const groupName = getGroupName(groupNameIndex);
            cell0.textContent = groupName;
    
            ['10min', '30min', '1h', '2h', '3h'].forEach((time, colIndex) => {
                const images = timeGroups[time];
                if (images.length > groupNameIndex && groupNameIndex < processedResults.length) { // 检查当前行是否有该时间段的图片
                    const cell = row.cells[colIndex + 1];
                    const img = document.createElement('img');
                    img.src = images[groupNameIndex].colored_output_path;
                    img.classList.add('image-cell');
                    img.onerror = function() {
                        cell.textContent = '图片加载失败';
                        console.error(`图片加载失败: ${img.src}`);
                    };
                    img.onload = function() {
                        console.log(`图片加载成功: ${img.src}`);
                    };
                    cell.appendChild(img);
                    cell.style.display = 'table-cell';
                }
            });
        }
    
        // 4. 隐藏没有图片的行
        for (let i = 0; i < 5; i++) {
            const hasImages = Array.from(tbody.rows[i].cells).some(cell => cell.style.display === 'table-cell');
            tbody.rows[i].style.display = hasImages ? '' : 'none';
        }
    
        // 5. 更新分页信息
        pageInfo1.textContent = `第 ${page + 1} 页 / 共 ${Math.ceil(processedResults.length / itemsPerPage)} 页`;
        prevBtn.disabled = page === 0;
        nextBtn.disabled = page >= Math.ceil(processedResults.length / itemsPerPage) - 1;
    }
    
    // 辅助函数：获取组名
    function getGroupName(index) {
        if (index < processedResults.length) {
            const result = processedResults[index];
            const match = result.colored_output_path.match(/(10min|30min|1h|2h|3h)_模型\.pth_colored_prediction_(\d+)/);
            if (match) {
                return `${match[1]} 组 ${parseInt(match[2], 10) + 1}`;
            }
        }
        return '未知组';
    }
    
    
    // 获取每组图片的最后一张图片的名称
    function getGroupName(index) {
        const groupSize = 6;
        const groupNameIndex = index + (groupSize - 1); // 每组的最后一张图片
        if (groupNameIndex < filenames.length) {
            const groupName = filenames[groupNameIndex].replace(/_模型\.pth_colored_prediction_\d+/, '');
            return groupName;
        } else {
            return (index + 1).toString();
        }
    }
    // 获取文件名列表
    function fetchFilenames() {
        fetch('/get_filenames')
            .then(response => response.json())
            .then(data => {
                filenames = data.filenames.map(filename => filename.split('.').slice(0, -1).join('.')); // 去掉文件扩展名
                updateImageNames(filenames);
            })
            .catch(error => logMessage('Error fetching filenames: ' + error));
    }
   // 更新表格中的文件名
    function updateImageNames(filenames) {
        const groupedNames = filenames.reduce((acc, filename, index) => {
            if ((index + 1) % 6 === 0) { // 每6个文件名取最后一个作为组名
                acc.push(filename.replace(/_模型\.pth_colored_prediction_\d+/, ''));
            }
            return acc;
        }, []);

        for (let i = 0; i < 5 && i < groupedNames.length; i++) {
            const cell = tbody.rows[i].cells[0];
            cell.textContent = groupedNames[i];
        }
    }
    // 分页按钮点击事件
    prevBtn.addEventListener('click', () => {
        if (currentPage > 0) {
            currentPage--;
            displayPage(currentPage);
        }
    });

    nextBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(processedResults.length / itemsPerPage);
        if (currentPage < totalPages - 1) {
            currentPage++;
            displayPage(currentPage);
        }
    });
    // 新增的函数，用于显示动画页面
    function displayAnimationPage(page) {
        console.log('Displaying animation page:', page);
    
        // 清空动画容器
        const animationContainer = document.getElementById('animation-container');
        animationContainer.innerHTML = '';
    
        const startIndex = page * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
    
        const pageResults = processedResults.slice(startIndex, endIndex);
    
        // 显示动画
        pageResults.forEach(result => {
            const img = document.createElement('img');
            img.src = result.gif_output_path;  // 假设每个结果都有一个 gif_output_path 属性
            img.classList.add('animation-image');
            animationContainer.appendChild(img);
        });    
        // 更新页数信息
        pageInfo2.textContent = `第 ${page + 1} 页 / 共 ${Math.ceil(processedResults.length / itemsPerPage)} 页`;
    
        // 更新按钮状态
        const prevAnimationBtn = document.getElementById('prev-animation');
        const nextAnimationBtn = document.getElementById('next-animation');
        prevAnimationBtn.disabled = page === 0;
        nextAnimationBtn.disabled = page >= Math.ceil(processedResults.length / itemsPerPage) - 1;
    }
    // 初始化动画分页按钮点击事件
    document.getElementById('prev-animation').addEventListener('click', () => {
        if (currentAnimationPage > 0) {
            currentAnimationPage--;
            displayAnimationPage(currentAnimationPage);
        }
    });

    document.getElementById('next-animation').addEventListener('click', () => {
        const totalPages = Math.ceil(processedResults.length / itemsPerPage);
        if (currentAnimationPage < totalPages - 1) {
            currentAnimationPage++;
            displayAnimationPage(currentAnimationPage);
        }
    });
    // 初始化显示第一页动画
    displayAnimationPage(currentAnimationPage);
});

function openFolder() {
    fetch('/open_folder', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            console.log(data.message); // 可选：在控制台中显示反馈信息
        }
    })
    .catch(error => console.error('Error:', error));
}

function openFolder() {
    fetch('/open_folder', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            console.log(data.message);
        }
    })
    .catch(error => console.error('Error opening folder:', error));
}
