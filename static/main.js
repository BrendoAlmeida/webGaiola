document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const statusElement = document.getElementById('button-status');

    socket.on('connect', () => {
        console.log('Conectado ao servidor via WebSocket!');
    });

    socket.on('button_status', function(msg) {
        console.log('Status do bebedouro recebido: ' + msg.data);

        if (msg.data.includes("DRINKING")) {
            statusElement.textContent = 'BEBENDO';
            statusElement.classList.remove('status-off');
            statusElement.classList.add('status-on');
        } else {
            statusElement.textContent = 'PARADO';
            statusElement.classList.remove('status-on');
            statusElement.classList.add('status-off');
        }
    });
});