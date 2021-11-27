from app import create_app
from flask_toastr import Toastr

app = create_app()
toastr = Toastr(app)

if __name__ == '__main__':
    from waitress import serve
    app.run(host='127.0.0.1', port='8080', threaded=True)
    serve(app)
