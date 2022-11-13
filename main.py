from flask import Flask, redirect, request, jsonify, render_template, make_response
import json
import os
from replit import db
from util import randomString
import requests
import random

app = Flask(__name__)


@app.route('/login')
def login():
    # state = randomString(16)
    # scope = [
    #     "user-read-playback-state", "app-remote-control",
    #     "user-modify-playback-state", "user-read-currently-playing",
    #     "user-read-playback-position", "user-read-email", "streaming"
    # ]
    # print('https://accounts.spotify.com/authorize?' + json.dumps({
    #   'response_type': 'code',
    #   'client_id': os.environ["client_id"],
    #   'scope':scope,
    #   'state':state
    # }
    # ))
    return redirect(f'https://accounts.spotify.com/en/authorize?client_id={os.environ['client_id']}&redirect_uri=https://scaredgrippingcalculators.ghelanibhavin.repl.co/auth/callback&response_type=code&scope=user-modify-playback-state')


@app.route('/auth/callback')
def authCallback():
    values = request.args
    if not values:
        response = {'message': 'No data found.'}
        return jsonify(response), 400
    if "code" not in values:
        response = {'message': 'Some data is missing'}
        return jsonify(response), 400
    db["token"] = values["code"]
    data = requests.post("https://accounts.spotify.com/api/token", headers={'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'Basic NmRkMjNmNDJhNmJkNGEyMzlmNWY3ZmYzZjM0NTU4ZTk6OTlhMGI1YWY4NGFlNDQwYWJhMWY5ZDc4ZDkxYzBhNWM='}, data={
                         "grant_type": "authorization_code", "code": db["token"], "redirect_uri": "https://scaredgrippingcalculators.ghelanibhavin.repl.co/auth/callback"})
    print(data.text)
    db["accessToken"] = json.loads(data.text)["access_token"]
    # args = request.args
    # db["tokenarg"] = args["code"]
    # print(args)

    resp = make_response(redirect('/'))
    resp.set_cookie('login', '1')
    return resp


@app.route('/play', methods=["POST"])
def play():
    values = request.get_json()
    if not values:
        response = {'message': 'No data found.'}
        return jsonify(response), 400
    if 'mood' not in values:
        response = {'message': 'Some data is missing'}
        return jsonify(response), 400
    no = random.randint(0, len(db[values["mood"]]) - 1)
    print(values['mood'])
    print(len(values['mood']))
    resp = requests.put("https://api.spotify.com/v1/me/player/play", headers={"Content-Type": "application/json", "Authorization": "Bearer " + db["accessToken"]}, json={
                        "position_ms": 1, "context_uri": f"spotify:playlist:{db[values['mood']][no]}"})
    print(resp.text)
    return jsonify({"playlist": db[values['mood']][no]}), 200


@app.route('/', methods=['POST', 'GET'])
def addPlaylist():
    if request.method == 'GET':
        if request.cookies.get('login') == '1':
            resp = make_response(render_template(
                'addplaylist.html', login="hidden"))
        else:
            resp = make_response(render_template(
                'addplaylist.html', loggedin="hidden"))
        return resp
    else:

        print(request.form.get("mood"))
        if request.form.get("pid") in db[request.form.get("mood")]:
            if request.cookies.get('login') == '1':
                resp = make_response(render_template(
                    'alreadyExists.html', login="hidden"))
                return resp
            else:
                resp = make_response(render_template(
                    'alreadyExists.html', loggedin="hidden"))
                return resp

        else:

            db[request.form.get("mood")].append(request.form.get("pid"))
            if request.cookies.get('login') == '1':
                resp = make_response(render_template(
                    'succAdd.html', login="hidden"))
                return resp
            else:
                resp = make_response(render_template(
                    'succAdd.html', loggedin="hidden"))
                return resp
            return render_template("succAdd.html")


@app.route('/test', methods=["POST"])
def test():
    return jsonify({"abc": "abc"})


app.run(host='0.0.0.0', port=8888)
