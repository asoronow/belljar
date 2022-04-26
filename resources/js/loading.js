var ipc = require('electron').ipcRenderer;

ipc.on('updateStatus', function(event, response){
    var status = document.getElementById('status');
    status.innerHTML = response;
});