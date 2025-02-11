document.getElementById("login-button").addEventListener("click", handleLogin);

async function handleLogin(event) {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
  
    console.log("Script triggered");

    const response = await fetch('http://localhost:5000/app/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
  
    console.log(await response.text());
    window.location.href = "home.html";
}