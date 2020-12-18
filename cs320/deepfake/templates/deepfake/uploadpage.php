<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
  <style>
    fieldset{
      height: 40px;
      width: 100%-4px;
      display: block;
      margin: 0 auto;
      border: 1px  solid #1f80aa;
      border-radius: 5px;
      backgroud: #ffffff

    }
    input{
      font-size: 16px;
      width: 100%-20px;
      padding: 10px;
      border: 0px;
      outline: none;
      float: left
    }
    button{
      width: 50px;
      height: 100%;
      border: 0px;
      background: #ffffff;
      outline: none;
      float: right;
    }
    li a{
      color: #000000
    }

    form2 {

      width: 250px;
      display: block;
      margin: 0 auto;
      border: 0px;
    }

  </style>
  <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>
</head>
   <script type="text/javascript">
        jQuery(document).ready(function($) {
            // hide the menu when the page load
            $("#navigation-list").hide();
            // when .menuBtn is clicked, do this
            $(".menuBtn").click(function() {
                // open the menu with slide effect
                $("#navigation-list").slideToggle(300);
            });
        });
    </script>
<body>
  <header>
    <div style="text-align:center"><h1><a href="index.html">Soccer Player</a></h1></div>

  </header>
  <header>
   <span class="menuBtn"><img src="ham.png" width="50"></span>
    <nav id="navigation-list">
      <ul>
        <li><a href="#item1">Soccer Player List</a></li>
        <li><a href="#item2">Goods shop</a></li>
        <li><a href="#item3">Settings</a></li>
      </ul>
    </nav>
  </header>
  <br>


  <br>

  <?php
  $conn = mysqli_connect('localhost', 'root', 'ab7103dfehgms!');
  mysqli_select_db($conn, 'soccerdeepfake');
  $playername = trim($_GET['playername']);
  $sql='SELECT * FROM playerlist WHERE name = "'.$playername.'"';
  $result = mysqli_query($conn,$sql);


  if ($result->num_rows > 0){
    $row = mysqli_fetch_assoc($result);
    echo
    '<div style="text-align:center">
        <img src='.$row['address'].' alt="error" width="300">
      </div>
      <br>

      <form2 action = "uploadpage.php" method="GET" enctype="multipart/form-data">
        <input value="파일선택" accept="image/*"/ disabled="disabled" style="width:200px">
          <label for="ex_filename">
            <img src="upload.png" alt="error" width="30" border="0" onclick="document.all.file.click()">
          </label>
        <input type="file" id="ex_filename" style="display:none">
      </form2>';
  } else { echo '<h3 style="text-align:center">not found</h3>'; }
  ?>




</body>
</html>
