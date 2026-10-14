"""PDF report generated from a real analysis result (ReportLab). Nothing is hard-coded."""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DISCLAIMER = ("Cutis Lens is an educational AI prototype, not a medical device. Its output is a prediction "
              "from a machine-learning model and must not be used for diagnosis or treatment decisions. "
              "Please consult a qualified dermatologist for any skin concern.")
LIMITATIONS = [
    "Trained on a single public dataset (ACNE04); performance on other cameras, lighting or skin tones is unknown.",
    "Confidence values are softmax scores, not calibrated probabilities of a medical condition.",
    "The heatmap (Grad-CAM) shows model attention, not proof that a lesion exists at that location.",
    "The model predicts an image-level severity category; it does not count or localise individual lesions.",
]


def build_report(result):
    """result keys: analysis_id, created_at, filename, predictions[list], explanation,
    model_info{}, images{'original','overlay','heatmap'} (PNG bytes). Returns PDF bytes."""
    for key in ("analysis_id", "predictions", "images", "explanation"):
        if key not in result:
            raise ValueError(f"result is missing '{key}'")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor("#3a3a9c"), fontSize=22)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#3a3a9c"), spaceBefore=8, spaceAfter=3, keepWithNext=1)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Cutis Lens analysis report", author="Cutis Lens")
    st = [Paragraph("Cutis Lens", h1),
          Paragraph("Explainable AI-Powered Acne Detection and Skin Analysis - Analysis Report", body),
          Spacer(1, 8)]
    meta = [["Analysis ID", result["analysis_id"]],
            ["Date", result.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Image", result.get("filename", "uploaded image")]]
    t = Table(meta, colWidths=[3.2 * cm, 13 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555")),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#ddd"))]))
    st += [t, Paragraph("Predicted categories and confidence", h2)]

    rows = [["Category", "Softmax score"]] + [[p["display"], f"{p['score']:.1%}"] for p in result["predictions"]]
    pt = Table(rows, colWidths=[9 * cm, 4 * cm])
    pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ece9fb")),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#ccc"))]))
    st += [pt, Paragraph("Explainability (Grad-CAM)", h2)]

    def img(key, width):
        from PIL import Image as PILImage
        w, h = PILImage.open(io.BytesIO(result["images"][key])).size
        return Image(io.BytesIO(result["images"][key]), width=width, height=width * h / w)

    gap = Table([[img("original", 6.6 * cm), img("overlay", 6.6 * cm)],
                 [Paragraph("Original image", small), Paragraph("Grad-CAM overlay", small)]],
                colWidths=[8.4 * cm, 8.4 * cm])
    st += [gap, Spacer(1, 6), Paragraph(result["explanation"], body), Paragraph("Model information", h2)]
    mi = result.get("model_info") or {}
    st.append(Paragraph("; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in mi.items()) or "not available", body))
    st.append(Paragraph("Important limitations", h2))
    st += [Paragraph(f"&bull; {x}", body) for x in LIMITATIONS]
    st += [Paragraph("Medical disclaimer", h2), Paragraph(DISCLAIMER, body)]
    doc.build(st)
    return buf.getvalue()
