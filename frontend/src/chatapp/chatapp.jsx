import React, { useState } from 'react';
// import ReactDOM from 'react-dom';
// import './index.css';

function ChatPage() {
    const [messages, setMessages] = useState([
        { text: 'Hello! How can I help you?', type: 'received' },
        { text: 'Hi, I need some information.', type: 'sent' }
    ]);
    const [input, setInput] = useState('');

    const sendMessage = () => {
        if (input.trim() !== '') {
            setMessages([...messages, { text: input, type: 'sent' }]);
            setInput('');
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-screen w-screen border border-gray-300">
            <div className="bg-emerald-500 text-white p-4 text-center text-lg">
                KUTCAS connect
            </div>
            <div className="flex-grow p-4 overflow-y-auto bg-gray-100">
                {messages.map((msg, index) => (
                    <div key={index} className={`flex items-center mb-3 ${msg.type === 'sent' ? 'justify-end' : 'justify-start'}`}>
                        {msg.type === 'received' && (
                            <div className="w-8 h-8 bg-gray-400 rounded-full flex items-center justify-center mr-2">
                                B
                            </div>
                        )}
                        <div
                            className={`p-3 rounded-full max-w-xs ${
                                msg.type === 'sent' ? 'bg-emerald-500 text-white' : 'bg-gray-300 text-black'
                            }`}
                        >
                            {msg.text}
                        </div>
                        {msg.type === 'sent' && (
                            <div className="w-8 h-8 bg-emerald-600 rounded-full flex items-center justify-center ml-2 text-white">
                                U
                            </div>
                        )}
                    </div>
                ))}
            </div>
            <div className="flex p-4 border-t border-gray-300 bg-white">
                <input
                    type="text"
                    className="flex-grow p-2 border border-gray-300 rounded-lg mr-2"
                    placeholder="Type a message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyPress}
                />
                <button
                    className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
                    onClick={sendMessage}
                >
                    Send
                </button>
            </div>
        </div>
    );
}

// ReactDOM.render(<ChatPage />, document.getElementById('root'));
export default ChatPage;