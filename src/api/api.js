
export const getAIMessage = async (userQuery) => {
  try {
    const response = await fetch('http://127.0.0.1:5000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json' 
      },
      body: JSON.stringify({ query: userQuery })
    });

    const data = await response.json();
    const message = { 
      role: "assistant",
      content: data.response
    };

    return message;
  } 
  catch (error) {
    console.error("Error:", error);
    return { content: "Error: Unable to get response from the agent." };
  }
};
