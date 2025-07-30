import 'dart:math';
import 'package:flutter/material.dart';

class CenterCircle extends StatelessWidget {
  const CenterCircle({required this.child, super.key});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: CustomPaint(
        size: const Size(160, 160),
        painter: NestedCirclesPainter(
          color: Theme.of(context).colorScheme.primary,
          strokeWidth: 1.0,
          gapBetweenCircles: 4.0,
        ),
        child: child,
      ),
    );
  }
}

// Custom Painter for drawing two nested circles
class NestedCirclesPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;
  final double gapBetweenCircles; // The space between the two circles

  NestedCirclesPainter({
    this.color = Colors.white54, // Default color for the circles
    this.strokeWidth = 1.5, // Default stroke width for both circles
    this.gapBetweenCircles = 4.0, // Default gap between the circles
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate the center of the drawing area
    final Offset center = Offset(size.width / 2, size.height / 2);

    // Configure the paint properties (same for both circles)
    final Paint paint = Paint()
      ..color = color
          .withValues(alpha: 0.7) // Make circles slightly transparent
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke; // Draw the outline

    // Calculate the radius for the outer circle
    // Ensure it fits within the bounds defined by 'size'
    final double outerRadius =
        min(size.width / 2, size.height / 2) - strokeWidth / 2;

    // Calculate the radius for the inner circle
    final double innerRadius =
        outerRadius - gapBetweenCircles - strokeWidth / 2;

    // Ensure inner radius is not negative
    if (innerRadius > 0) {
      // Draw the outer circle
      canvas.drawCircle(center, outerRadius, paint);
      // Draw the inner circle
      canvas.drawCircle(center, innerRadius, paint);
    } else {
      // If the gap is too large, just draw the outer circle
      canvas.drawCircle(center, outerRadius, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    // Repaint only if properties change
    return oldDelegate is NestedCirclesPainter &&
        (oldDelegate.color != color ||
            oldDelegate.strokeWidth != strokeWidth ||
            oldDelegate.gapBetweenCircles != gapBetweenCircles);
  }
}
