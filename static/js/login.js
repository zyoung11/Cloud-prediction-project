document.addEventListener('DOMContentLoaded', function() {
    const login = document.querySelector('.login');
    let span;
    let inTime, outTime;
    let isIn = true;
    let isOut;
    const loginForm = document.getElementById('loginForm');
    const registerButton = document.getElementById('registerButton');
    const enterButton = document.getElementById('enterButton');

    // 鼠标进入事件
    login.addEventListener('mouseenter', function(e) {
        isOut = false;
        if (isIn) {
            inTime = new Date().getTime();
            span = document.createElement('span');
            login.appendChild(span);

            span.style.animation = 'in .5s ease-out forwards';
            const top = e.clientY - login.offsetTop;
            const left = e.clientX - login.offsetLeft;

            span.style.top = `${top}px`;
            span.style.left = `${left}px`;

            isIn = false;
            isOut = true;
        }
    });

    // 鼠标离开事件
    login.addEventListener('mouseleave', function(e) {
        if (isOut) {
            outTime = new Date().getTime();
            const passTime = outTime - inTime;

            if (passTime < 500) {
                setTimeout(mouseleave, 500 - passTime);
            } else {
                mouseleave();
            }
        }

        function mouseleave() {
            span.style.animation = 'out .5s ease-out forwards';
            const top = e.clientY - login.offsetTop;
            const left = e.clientX - login.offsetLeft;

            span.style.top = `${top}px`;
            span.style.left = `${left}px`;

            setTimeout(function() {
                login.removeChild(span);
                isIn = true;
            }, 500);
        }
    });

    // 表单提交事件处理程序
    loginForm.addEventListener('submit', function(event) {
        event.preventDefault(); // 阻止表单默认提交行为

        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value.trim();

        // 检查账号和密码的输入情况，并给出相应提示
        if (!username || !password) {
            alert('请输入账号和密码');
            enterButton.disabled = true; // 保持按钮禁用状态
            return;
        }

        // 禁用按钮以防止重复提交
        enterButton.disabled = true;

        // 发送POST请求进行登录验证
        fetch('/login_post', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                'username': username,
                'password': password
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.message === '登录成功！') {
                alert('登录成功');
                window.location.href = '/data_mode'; // 替换为你要跳转的目标URL
            } else {
                alert('账号或密码错误');
            }
            enterButton.disabled = false; // 重新启用按钮
        })
        .catch(error => {
            console.error('Error:', error);
            enterButton.disabled = false; // 重新启用按钮
        });
    });

    // 注册功能
    registerButton.addEventListener('click', function(event) {
        event.preventDefault();

        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value.trim();

        if (!username || !password) {
            alert('请输入账号和密码');
            return;
        }

        // 禁用按钮以防止重复提交
        registerButton.disabled = true;

        // 发送POST请求进行注册
        fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                'username': username,
                'password': password
            })
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            if (data.message === '注册成功！') {
                registerButton.disabled = false; // 注册成功后启用“注册”按钮
            }
        })
        .catch(error => {
            console.error('Error:', error);
            registerButton.disabled = false; // 重新启用按钮
        });
    });

    // 初始化时禁用“注册”按钮
    enterButton.disabled = true;
});