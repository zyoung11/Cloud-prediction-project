document.addEventListener('DOMContentLoaded', function() {    
    const fileInput = document.getElementById('file-input');    
    const uploadForm = document.getElementById('upload-form');    
    const uploadBtn = document.getElementById('upload-btn');    
    const processBtn = document.getElementById('process-btn');    
    const logDiv = document.getElementById('log');    
    const resultsDiv = document.getElementById('results');    

    let uploadedFiles = [];    
    let groupedResults = []; // 用于存储分组后的结果    
    const altLabels = ['10min', '30min', '1h', '2h', '3h',     
                       '10min', '30min', '1h', '2h', '3h',       
                       '10min', '30min', '1h', '2h', '3h',   
                       '10min', '30min', '1h', '2h', '3h',  
                        '10min', '30min', '1h', '2h', '3h'];   
  
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
                groupedResults = groupAndOrganizeResults(data.results);    
                displayResults();    
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
    function groupAndOrganizeResults(results) {    
        let grouped = [];  
        for (let i = 0; i < results.length; i += 5) {  
            grouped.push(results.slice(i, i + 5));  
        }  
        return grouped;  
    } 

    function displayResults() {    
        resultsDiv.innerHTML = '';    
        resultsDiv.style.display = 'flex';    
        resultsDiv.style.flexDirection = 'column';    
        
        // 获取特定图片的名称（这里假设是日期名称）    
        const specificIndices = [5, 11, 17, 23, 29]; // 注意索引从0开始    
        const specificDates = specificIndices.map(index => uploadedFiles[index - 1]); // 获取对应的日期名称    
        
        let groupIndex = 0; // 用于跟踪当前处理的是第几组    
        let dateContainers = []; // 存储日期容器的数组，以便后续使用  
        
        groupedResults.forEach(group => {    
            let groupContainer = document.createElement('div');    
            groupContainer.style.display = 'grid';    
            groupContainer.style.gridTemplateColumns = 'repeat(5, 1fr)'; // 5列，每列等宽    
            groupContainer.style.gap = '10px'; // 网格项之间的间距（可以根据需要调整）    
        
            // 如果这是第一组，则创建日期容器  
            if (groupIndex === 0) {    
                specificDates.forEach((date, idx) => {    
                    let dateContainer = document.createElement('div');    
                    dateContainer.textContent = date;    
                    dateContainer.style.marginTop = '5px';    
                    dateContainer.style.textAlign = 'center';    
                    dateContainer.style.fontSize = '14px';    
                    dateContainer.style.fontWeight = 'bold';    
                    dateContainers.push(dateContainer); // 存储日期容器  
        
                    // 立即将日期容器添加到正确的位置（每行的第一个位置）  
                    let placeholder = document.createElement('div'); // 占位符，用于保持布局  
                    groupContainer.appendChild(placeholder);  
                });    
            }  
  
            // 为当前组的每个结果创建图片容器，并替换占位符（如果是第一组）  
            group.forEach((result, resultIdx) => {    
                let itemContainer = document.createElement('div');    
                itemContainer.style.display = 'flex';    
                itemContainer.style.flexDirection = 'column';    
                itemContainer.style.alignItems = 'center';    
        
                const img = document.createElement('img');    
                img.src = '/' + result.path;
                img.alt = altLabels[groupedResults.flat().indexOf(result)] || 'Prediction result';     
                img.alt = 'Prediction result';    
                img.style.width = "200px";    
                img.style.height = "auto";
                

                const modelNameDiv = document.createElement('div');      
                modelNameDiv.textContent = 'Model: ' + altLabels[groupedResults.flat().indexOf(result) % altLabels.length];
        
                if (groupIndex === 0) {  
                    // 替换第一行的占位符为日期容器  
                    let placeholder = groupContainer.children[resultIdx];  
                    groupContainer.replaceChild(dateContainers[resultIdx], placeholder);  
                }  
        
                itemContainer.appendChild(img);
                itemContainer.appendChild(modelNameDiv);    
                groupContainer.appendChild(itemContainer); // 总是添加图片容器  
            });    
            resultsDiv.appendChild(groupContainer);    
            groupIndex++;    
        });    
    }  
}); 