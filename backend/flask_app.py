from flask import Flask, request, jsonify
from part_select_agent import PartSelectAgent

app = Flask(__name__)

llm_agent = PartSelectAgent()

@app.route('/chat', methods=['POST'])
def chat():
    if request.is_json:
        data = request.get_json()
        user_query = data.get('query', '') 
        agent_response = llm_agent.handle_message(user_query)
        return jsonify({"response": agent_response})
    else:
        return jsonify({"error": "Invalid request format, JSON expected"}), 400

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
