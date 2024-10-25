const itemsPerPage = 1; // 每页显示的 GIF 数量
let currentPage = 1;
let totalItems = 0;
let gifData = null;

async function fetchGIFs() {
    const response = await fetch('/get_gifs');
    const newData = await response.json();

    // 如果有新的GIF生成，更新gifData和totalItems
    if (newData.gifs.length > totalItems) {
        gifData = newData;
        totalItems = gifData.gifs.length;
        displayGIFs(); // 更新显示
    }
}

function displayGIFs() {
    if (!gifData || !gifData.gifs) return;

    const index = currentPage - 1;
    const gifs = document.getElementById('animation-container');
    gifs.innerHTML = '';

    if (index < totalItems) {
        const gif = document.createElement('img');
        gif.src = `/outputs/gif/${gifData.gifs[index]}`;
        gifs.appendChild(gif);
    }

    document.getElementById('prev-animation').disabled = currentPage === 1;
    document.getElementById('next-animation').disabled = currentPage === totalItems;
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('prev-animation').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            displayGIFs();
        }
    });

    document.getElementById('next-animation').addEventListener('click', () => {
        if (currentPage < totalItems) {
            currentPage++;
            displayGIFs();
        }
    });

    fetchGIFs(); // 初始加载GIF

    // 设置一个定时器，每5秒钟检查一次新的GIF
    setInterval(fetchGIFs, 5000); // 5000毫秒 = 5秒
});