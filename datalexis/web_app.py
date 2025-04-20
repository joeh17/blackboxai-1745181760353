from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import plotly.express as px
import io
import os
import tempfile

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

data = None
data_path = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global data, data_path
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file:
            data_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(data_path)
            data = pd.read_csv(data_path)
            return redirect(url_for('overview'))
    return render_template('index.html')

@app.route('/overview')
def overview():
    if data is None:
        return redirect(url_for('index'))
    summary = data.describe().to_html()
    head = data.head().to_html()
    return render_template('overview.html', summary=summary, head=head)

@app.route('/visualization')
def visualization():
    if data is None:
        return redirect(url_for('index'))
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        return render_template('visualization.html', error="Not enough numeric columns for scatter plot.")
    fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")
    graph_html = fig.to_html(full_html=False)
    return render_template('visualization.html', plot=graph_html)

@app.route('/download_report')
def download_report():
    if data is None:
        return redirect(url_for('index'))
    temp_dir = tempfile.mkdtemp()
    report_path = os.path.join(temp_dir, "datalexis_report.html")

    summary = data.describe().to_html()
    head = data.head().to_html()
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    fig_html = ""
    if len(numeric_cols) >= 2:
        fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")
        fig_html = fig.to_html(full_html=False)

    html_content = f"""
    <html>
    <head>
        <title>Datalexis Analysis Report</title>
        <link rel="icon" type="image/svg+xml" href="{{{{ url_for('static', filename='favicon.svg') }}}}">
    </head>
    <body>
    <h1>Datalexis Analysis Report</h1>
    <h2>Data Head</h2>
    {head}
    <h2>Summary Statistics</h2>
    {summary}
    <h2>Scatter Plot</h2>
    {fig_html}
    </body>
    </html>
    """

    with open(report_path, 'w') as f:
        f.write(html_content)

    return send_file(report_path, as_attachment=True, download_name="datalexis_report.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
