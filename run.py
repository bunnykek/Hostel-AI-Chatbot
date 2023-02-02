from flask import Flask, request, jsonify
from flask_cors import CORS

from chatbot import Chatbot

app = Flask(__name__)
CORS(app)

bot = Chatbot()

@app.route('/api/v1/answer', methods=['POST'])
def getResponse():
    data = request.get_json()
    print(data)
    try:
        query = data['query']
        queryAns = bot.response(query)
        print(queryAns)
        if not queryAns:
            return jsonify({'query': query,
                        'answer': "Sorry, we couldn't resolve your query. This query will be used for the further improvement of the bot."}), 201
        return jsonify({'query': query,
                        'answer': queryAns}), 201
    except:
        return jsonify({'message': 'Some error occurred!'}), 401

if __name__ == '__main__':
    app.run(debug=True)