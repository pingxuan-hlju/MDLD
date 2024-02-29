<html lang="en"><head>
    <meta charset="UTF-8">
    <title></title>
<style id="system" type="text/css">h1,h2,h3,h4,h5,h6,p,blockquote {    margin: 0;    padding: 0;}body {    font-family: "Helvetica Neue", Helvetica, "Hiragino Sans GB", Arial, sans-serif;    font-size: 13px;    line-height: 18px;    color: #737373;    margin: 10px 13px 10px 13px;}a {    color: #0069d6;}a:hover {    color: #0050a3;    text-decoration: none;}a img {    border: none;}p {    margin-bottom: 9px;}h1,h2,h3,h4,h5,h6 {    color: #404040;    line-height: 36px;}h1 {    margin-bottom: 18px;    font-size: 30px;}h2 {    font-size: 24px;}h3 {    font-size: 18px;}h4 {    font-size: 16px;}h5 {    font-size: 14px;}h6 {    font-size: 13px;}hr {    margin: 0 0 19px;    border: 0;    border-bottom: 1px solid #ccc;}blockquote {    padding: 13px 13px 21px 15px;    margin-bottom: 18px;    font-family:georgia,serif;    font-style: italic;}blockquote:before {    content:"C";    font-size:40px;    margin-left:-10px;    font-family:georgia,serif;    color:#eee;}blockquote p {    font-size: 14px;    font-weight: 300;    line-height: 18px;    margin-bottom: 0;    font-style: italic;}code, pre {    font-family: Monaco, Andale Mono, Courier New, monospace;}code {    background-color: #fee9cc;    color: rgba(0, 0, 0, 0.75);    padding: 1px 3px;    font-size: 12px;    -webkit-border-radius: 3px;    -moz-border-radius: 3px;    border-radius: 3px;}pre {    display: block;    padding: 14px;    margin: 0 0 18px;    line-height: 16px;    font-size: 11px;    border: 1px solid #d9d9d9;    white-space: pre-wrap;    word-wrap: break-word;}pre code {    background-color: #fff;    color:#737373;    font-size: 11px;    padding: 0;}@media screen and (min-width: 768px) {    body {        width: 748px;        margin:10px auto;    }}</style><style id="custom" type="text/css"></style></head>
<body marginheight="0"><h1>MDLD</h1>
<h2>Introduction</h2>
<p>This is code of MDLD (“Two enhanced features to predict the disease-associated lncRNAs”).

</p>
<h2>Dataset</h2>
<p>| File_name                  | Data_type       | Source                                                       |
| -------------------------- | --------------- | ------------------------------------------------------------ |
| dis_sim_matrix_process.txt | disease-disease | <a href="https://www.nlm.nih.gov/mesh/meshhome.html">MeSH</a>           |
| lnc_dis_association.txt    | lncRNA-disease  | <a href="https://www.cuilab.cn/lncrnadisease">LncRNADisease</a>         |
| mi_dis.txt                 | miRNA-disease   | <a href="https://www.cuilab.cn/hmdd">HMDD</a>                           |
| lnc_mi.txt                 | lncRNA-miRNA    | <a href="https://rnasysu.com/encori/">starBase</a>                      |
| lnc_sim.txt                | lncRNA-lncRNA   | <a href="https://www.nature.com/articles/srep11338">Chen <em>et al.</em></a>$^{1}$ |

</p>
<p>(1) Chen, X., Clarence Yan, C., Luo, C. <em>et al.</em> Constructing lncRNA Functional Similarity Network Based on lncRNA-Disease Associations and Disease Semantic Similarity. <em>Sci Rep</em> <strong>5</strong>, 11338 (2015).

</p>
<h1>File</h1>
<pre><code class="lang-markdown">-utils : data preprocessing,parameters,experimental evaluation
-data : data set
-models : build model
-main : model training and test</code></pre>
<h2>Environment</h2>
<pre><code class="lang-markdown">packages:
python == 3.9.0
torch == 1.13.0
numpy == 1.23.5
scikit-learn == 1.2.2
scipy == 1.10.1
pandas == 2.0.1
matplotlib == 3.7.1</code></pre>
<h1>Run</h1>
<pre><code class="lang-python">python ./main.py</code></pre>
<p>Edit By <a href="http://mahua.jser.me">MaHua</a>
Edit By <a href="http://mahua.jser.me">MaHua</a></p>
</body></html>