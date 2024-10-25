document.addEventListener('DOMContentLoaded', function() {    
    const fileInput = document.getElementById('file-input');    
    const uploadForm = document.getElementById('upload-form');    
    const uploadBtn = document.getElementById('upload-btn');    
    const processBtn = document.getElementById('process-btn');    
    const logDiv = document.getElementById('log');    
    const resultsTable = document.getElementById('results-table');    
    const tbody = resultsTable.querySelector('tbody');    
    const prevBtn = document.getElementById('prev-btn');    
    const nextBtn = document.getElementById('next-btn');    
    const pageInfo = document.getElementById('page-info');    
    let uploadedFiles = [];    
    let processedResults = [];    
    let currentPage = 0;    
    const itemsPerPage = 25;    

    fileInput.addEventListener('change', function() {    
        uploadBtn.disabled = this.files.length === 0;    
    });    

    uploadForm.addEventListener('submit', function(e) {    
        e.preventDefault();    
        const formData = new FormData();    
        for (let file of fileInput.files) {    
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
                uploadedFiles = Array.from(fileInput.files).map(file => file.name);    
                processBtn.disabled = false;    
            }    
        })    
        .catch(error => logMessage('Error uploading file: ' + error));    
    });    

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
                processedResults = data.results;    
                displayPage(currentPage);    
            }    
        })    
        .catch(error => logMessage('Error processing files: ' + error));    
    });    

    function logMessage(message) {    
        const p = document.createElement('p');    
        p.textContent = message;    
        logDiv.appendChild(p);    
        logDiv.scrollTop = logDiv.scrollHeight;    
    }    

    function displayPage(page) {    
        tbody.innerHTML = '';    
        const startIndex = page * itemsPerPage;    
        const endIndex = startIndex + itemsPerPage;    
        const pageResults = processedResults.slice(startIndex, endIndex);    

        // 初始化表格行
        for (let i = 1; i < 6; i++) {
            const row = document.createElement('tr');
            const cell0 = document.createElement('td');
            cell0.textContent = i.toString();
            row.appendChild(cell0);
            for (let j = 0; j < 5; j++) {
                const cell = document.createElement('td');
                row.appendChild(cell);
            }
            tbody.appendChild(row);
        }

        pageResults.forEach((result, index) => {      
            const row = Math.floor(index / 5);      
            const col = index % 5;      
            const cell = tbody.rows[row].cells[col + 1];      
            const img = document.createElement('img');      
            img.src = result.path;  // 使用 result.path 而不是 '/outputs/' + result.path
            img.classList.add('image-cell');  
            img.onerror = function() {  
                console.error('Failed to load image:', img.src);  // 添加错误处理  
            };  
            cell.appendChild(img);      
            if (col === 0 && result.inputFileName) {  
                tbody.rows[row].cells[0].textContent = result.inputFileName.split('.')[0];      
            }      
        });

        pageInfo.textContent = `第 ${page + 1} 页 / 共 ${Math.ceil(processedResults.length / itemsPerPage)} 页`;    
        prevBtn.disabled = page === 0;    
        nextBtn.disabled = page >= Math.ceil(processedResults.length / itemsPerPage) - 1;    
    }    

    prevBtn.addEventListener('click', () => {    
        if (currentPage > 0) {    
            currentPage--;    
            displayPage(currentPage);    
        }    
    });    

    nextBtn.addEventListener('click', () => {    
        if (currentPage < Math.ceil(processedResults.length / itemsPerPage) - 1) {    
            currentPage++;    
            displayPage(currentPage);    
        }    
    });    
});