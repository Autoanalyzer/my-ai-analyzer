document.getElementById('login-form').addEventListener('submit', async function(event) {
    event.preventDefault(); // ป้องกันการโหลดหน้าใหม่

    const username = event.target.username.value;
    const password = event.target.password.value;
    const errorMessage = document.getElementById('error-message');

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();

        if (response.ok) {
            // ถ้า Login สำเร็จ ให้ไปยังหน้าหลัก
            window.location.href = '/index.html';
        } else {
            // ถ้าไม่สำเร็จ ให้แสดงข้อความผิดพลาด
            errorMessage.textContent = data.error || 'Login failed!';
            errorMessage.style.display = 'block';
        }
    } catch (error) {
        errorMessage.textContent = 'Could not connect to the server.';
        errorMessage.style.display = 'block';
    }
});