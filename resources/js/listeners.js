var ipc = require('electron').ipcRenderer;
var indir = document.getElementById('indir');
var outdir = document.getElementById('outdir');
indir.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'indir') {
            indir.value = response[0];
        }
    })
    ipc.send('openDialog', 'indir');
});

outdir.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'outdir') {
            outdir.value = response[0];
        }
    })
    ipc.send('openDialog', 'outdir');
});