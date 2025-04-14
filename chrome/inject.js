if (!document.getElementById("my-sidebar")) {

    const sidebar = document.createElement("div");
    sidebar.id = "my-sidebar";
    sidebar.innerHTML = `<div id="welcome-message"> Welcome to ClosetX </div>`;
    
    document.body.appendChild(sidebar);
  
    const style = document.createElement("style");
    style.textContent = `
      #my-sidebar {
        position: fixed;
        color: black;
        background; solid;
        background-color: white;
        top: 0;
        right: 0;
        width: 300px;
        height: 100%;
        z-index: 10000;
        }
      #welcome-message {
        opacity: 0;
        animation: fadeIn 2s ease forwards;
        text-align: center;
        font-size: 1.5rem;
        font-family: 'IBM Plex Mono';
        padding-top: 1rem;
        }
      @keyframes fadeIn {
        to {
          opacity: 1;
        }
    `;
    document.head.appendChild(style);

    fetch("https://api.quotable.io/random")  // This returns a random quote
    .then(res => res.json())
    .then(data => {
      const container = document.getElementById("welcome-message");
      container.insertAdjacentHTML("beforeend", `
        <div style="margin-top: 15px;">
          <p>"${data.content}"</p>
          <p style="font-size: 0.9em; color: gray;">— ${data.author}</p>
        </div>
      `);
    });
}  