import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils import store_user_data, verify_user_data, load_image, record_current_time, move_folders_to_new_directory, npy_to_png
from models import run_inference, Generator, load_model
import webbrowser,subprocess

app = Flask(__name__)

# 定义用户数据文件路径
USER_DATA_FILE = 'Login.txt'
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
IMATEST_FOLDER = 'imatest'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'npy'}

model_files = [
    ("10min_模型.pth", {"input_channels": 3, "hidden_units": 64, "num_layers": 4}),
    ("30min_模型.pth", {"input_channels": 3, "hidden_units": 64, "num_layers": 4}),
    ("1h_模型.pth", {"input_channels": 3, "hidden_units": 64, "num_layers": 4}),
    ("2h_模型.pth", {"input_channels": 3, "hidden_units": 64, "num_layers": 4}),
    ("3h_模型.pth", {"input_channels": 3, "hidden_units": 64, "num_layers": 4})
]

models = {
    "10min_模型.pth": load_model("10min_模型.pth", Generator, 1, 64, 4),
    "30min_模型.pth": load_model("30min_模型.pth", Generator, 1, 64, 4),
    "1h_模型.pth": load_model("1h_模型.pth", Generator, 1, 64, 4),
    "2h_模型.pth": load_model("2h_模型.pth", Generator, 1, 64, 4),
    "3h_模型.pth": load_model("3h_模型.pth", Generator, 1, 64, 4)
}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
if not os.path.exists(IMATEST_FOLDER):
    os.makedirs(IMATEST_FOLDER)

app.config['WERKZEUG_SERVER_REQUEST_MAX_SIZE'] = 1024 * 1024 * 1024  # 设置为1GB  
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['IMATEST_FOLDER'] = 'imatest'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'npy'} 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMATEST_FOLDER'], exist_ok=True)



@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET'])  # 登录页面
def login():
    return render_template('login.html')#render_template 用于渲染HTML模板。

# 文件处理和推理相关功能
@app.route('/data_mode')
def data_mode():
    return render_template('data_mode.html')

@app.route('/show')
def show():
    return render_template('show.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('file')
    if not files:
        return jsonify({'error': 'No selected files'}), 400

    valid_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            valid_files.append(file_path)

    if not valid_files:
        return jsonify({'error': 'No valid files uploaded'}), 400

    return jsonify({'message': 'Files uploaded successfully', 'file_count': len(valid_files)}), 200

@app.route('/register', methods=['POST'])#注册
def register():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({'message': '请输入账号和密码'}), 400

    store_user_data(username, password)
    return jsonify({'message': '注册成功！'}), 200

@app.route('/login_post', methods=['POST'])#登录
def login_post():
    data = request.form
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({'message': '请输入账号和密码'}), 400

    if verify_user_data(username, password):
        return jsonify({'message': '登录成功！'}), 200
    else:
        return jsonify({'message': '账号或密码错误'}), 401
    
@app.route('/get_filenames', methods=['GET'])
def get_filenames():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    filenames = [f for f in files if allowed_file(f)]
    npy_to_png("uploads")
    return jsonify({'filenames': filenames}), 200

@app.route('/process', methods=['POST'])  
def process_files():
    # Run inference when this route is called
    load_image("uploads")
    results = run_inference(models, model_files, upload_folder='uploads', output_folder='outputs')
    return jsonify({'results': results}), 200

@app.route('/get_gifs', methods=['GET'])
def get_gifs():
    gif_folder = os.path.join('outputs', 'gif')
    if not os.path.exists(gif_folder):
        return jsonify({'message': 'GIF folder not found'}), 404

    gifs = [f for f in os.listdir(gif_folder) if f.endswith('.gif')]
    return jsonify({'gifs': gifs}), 200


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/outputs/gif/<path:filename>')
def serve_gif(filename):
    return send_from_directory(os.path.join('outputs', 'gif'), filename)

@app.route('/imatest/<path:filename>')
def serve_image(filename):
    return send_from_directory('imatest', filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 设置输出文件夹路径
OUTPUT_FOLDER_PATH = os.path.abspath('outputs')

@app.route('/open_folder', methods=['POST'])
def open_folder():
    if os.name == 'nt':  # Windows
        os.startfile(OUTPUT_FOLDER_PATH)
    elif os.name == 'posix':  # macOS or Linux
        subprocess.Popen(['open', OUTPUT_FOLDER_PATH])  # macOS
    else:
        subprocess.Popen(['xdg-open', OUTPUT_FOLDER_PATH])  # Linux
    return jsonify({"message": "Folder opened"}), 200


#区块图片处理相关功能
@app.route('/api/imatest/<folder>', methods=['GET'])
def get_images(folder):
    folder_path = os.path.join(IMATEST_FOLDER, folder)
    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder not found"}), 404
    
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return jsonify(images)

if __name__ == '__main__':
    move_folders_to_new_directory()
    record_current_time()
    local_server_url = "http://127.0.0.1:5000/"
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        webbrowser.open(local_server_url)
    app.run(debug=True)






   