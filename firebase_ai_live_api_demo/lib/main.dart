import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:firebase_core/firebase_core.dart';
import './firebase_options.dart';
import 'src/flutterfire_ai_live_api_demo.dart';
import './src/ui_components/ui_components.dart';

FirebaseOptions options = DefaultFirebaseOptions.currentPlatform;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: options);

  runApp(const ProviderScope(child: MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter x Firebase AI Live API Demo',
      home: FlutterFireAILiveAPIDemo(), // Demo app here
      theme: themeData,
      debugShowCheckedModeBanner: false,
    );
  }
}
