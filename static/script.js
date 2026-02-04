$(document).ready(() => {
    const chatBox = $("#messageFormeight");
    const scroll = () => chatBox.animate({ scrollTop: chatBox[0].scrollHeight }, 500);

    const appendMsg = (text, isUser) => {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const content = isUser ? text : marked.parse(text);
        const html = `
            <div class="d-flex ${isUser ? 'justify-content-end' : 'justify-content-start'} mb-4">
                <div class="${isUser ? 'msg_cotainer_send' : 'msg_cotainer'}">
                    ${content} <span class="msg_time" style="${isUser ? 'right:0' : 'left:0'}">${time}</span>
                </div>
            </div>`;
        chatBox.append(html);
        if (!isUser) chatBox.find('pre code').each((i, b) => hljs.highlightElement(b));
        scroll();
    };

    $("#messageArea").on("submit", (e) => {
        e.preventDefault();
        const msg = $("#text").val().trim();
        if (!msg) return;

        appendMsg(msg, true);
        $("#text").val("").css("height", "50px");

        const loader = $('<div id="loading" class="mb-4 typing-indicator"><span></span><span></span><span></span></div>');
        chatBox.append(loader);
        scroll();

        $.post("/get", { msg }).done((data) => {
            $("#loading").remove();
            appendMsg(typeof data === 'string' ? data : JSON.stringify(data), false);
        });
    });
});