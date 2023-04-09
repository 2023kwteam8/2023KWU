import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(Fixer());
}

class Fixer extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fixer',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomePage(),
    );
  }
}


class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}


class _HomePageState extends State<HomePage> {
  File _image;

  Future<void> _getImage(ImageSource source) async {
    final image = await ImagePicker().getImage(source: source);
    setState(() {
      _image = File(image.path);
    });
  }

  void _invertColors() async {
    final bytes = await _image.readAsBytes();
    final image = img.decodeImage(bytes);
    img.invertColors(image);
    final newImage = await File('after.jpg').writeAsBytes(img.encodeJpg(image));
    print('Saved image to: ${newImage.path}');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Fixer'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            if (_image != null) Image.file(_image),
            SizedBox(height: 16.0),
            RaisedButton(
              child: Text('촬 영'),
              onPressed: () => _getImage(ImageSource.camera),
            ),
            SizedBox(height: 16.0),
            RaisedButton(
              child: Text('색상 반전'),
              onPressed: _image != null ? _invertColors : null,
            ),
          ],
        ),
      ),
    );
  }
}
