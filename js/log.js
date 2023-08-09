var ipc = require('electron').ipcRenderer;

var log = document.getElementById('log');


function makeSpan(text){
    // Make lines separeted by <br> and add span
    return `<br></br><span>${text}</span>`;
}

function cacheLog(){
    // store log between refreshes
    var log = document.getElementById('log');
    var logText = log.innerHTML;
    localStorage.setItem('log', logText);
}

function loadLog(){
    // load log from cache
    var log = document.getElementById('log');
    var logText = localStorage.getItem('log');
    log.innerHTML = logText;
}

function clearCache(){
    // clear log cache
    localStorage.removeItem('log');
}

// on load
window.onload = function(){
    loadLog();
}

ipc.on('log', function(event, response){
    console.log(response);
    // Insert log message as span in log
    // Split response text by newlines
    let lines = response.split('\n');
    // Make each line a span
    lines.forEach(function(line){
        log.innerHTML += makeSpan(line);
    });
    // Scroll to bottom of log
    window.scrollTo(0,document.body.scrollHeight);

    cacheLog();
});