var ipc = require('electron').ipcRenderer;

ipc.on('updateStatus', function(event, response){
    var status = document.getElementById('status');
    status.innerHTML = response;
});

// get version from ipc
ipc.send('getVersion');
ipc.on('version', function(event, response){
    var version = document.getElementById('version');
    version.innerHTML = response;
});
