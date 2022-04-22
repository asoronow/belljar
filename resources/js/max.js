var ipc = require('electron').ipcRenderer;
var run = document.getElementById('run');
var indir = document.getElementById('indir');
var outdir = document.getElementById('outdir');

run.addEventListener('click', function(){
    ipc.once('maxResult', function(event, response){
        $('#processing').modal('hide')
        $('#done').modal('show')
    });

    ipc.once('maxError', function(event, response){
        $('#processing').modal('hide')
        $('#error').modal('show')
    });


    if (indir && outdir && indir.value && outdir.value) {
        ipc.send('runMax', [indir.value, outdir.value]);
    }
});
