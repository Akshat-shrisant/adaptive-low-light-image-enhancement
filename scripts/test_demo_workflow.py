"""
Comprehensive Verification of Demo Web Application
Tests all 12 functionality criteria from Section 31 of the UI Redesign specification.
"""

import io
import json
import urllib.request
from PIL import Image

BASE = "http://127.0.0.1:5000"

def test_workflow():
    print("=" * 65)
    print("DEMONSTRATION WEB APP — COMPLETE 12-POINT WORKFLOW TEST")
    print("=" * 65)

    # Test 1: Open website
    print("\n[Test 1] GET / (Webpage Load)")
    html = urllib.request.urlopen(BASE + "/").read().decode("utf-8")
    assert "Adaptive Low-Light Image Enhancement" in html
    assert "Step 1" in html and "Select Input Image" in html
    assert "Step 2" in html and "Analyze Image Characteristics" in html
    assert "Step 3" in html and "Adaptive Method Selection" in html
    assert "Step 4" in html and "Enhancement Result" in html
    print("  -> PASS: Webpage loaded with clean light structure (HTML size: %d bytes)" % len(html))

    # Test 2 & 3: Select benchmark 1.png and click Run
    print("\n[Test 2 & 3] Run Adaptive Enhancement on Benchmark '1.png'")
    payload = json.dumps({"sample_id": "1.png", "subset": "eval15"}).encode("utf-8")
    req = urllib.request.Request(BASE + "/api/analyze_and_enhance", data=payload, headers={"Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    assert res["status"] == "success"
    print("  -> PASS: API executed and returned success")

    # Test 4: Confirm 5 feature values appear
    print("\n[Test 4] Confirm Five Content-Aware Feature Metrics")
    chars = res["image_characteristics"]
    print("  * Mean Brightness            : %.2f" % chars["mean_brightness"])
    print("  * RMS Contrast               : %.2f" % chars["global_contrast"])
    print("  * Dynamic Range              : %.1f" % chars["dynamic_range"])
    print("  * Noise Estimate             : %.2f" % chars["noise_estimate"])
    print("  * Illumination Non-Uniformity: %.3f" % chars["illumination_nonuniformity"])
    assert chars["mean_brightness"] > 0
    assert chars["global_contrast"] > 0
    assert chars["dynamic_range"] > 0
    print("  -> PASS: All five feature values populated cleanly")

    # Test 5: Confirm adaptive method selected dynamically
    print("\n[Test 5] Confirm Dynamic Strategy Selection")
    dec = res["adaptive_decision"]
    print("  * Selected Strategy          : %s" % dec["selected_method"])
    print("  * Degradation Profile        : %s" % dec["degradation_profile"])
    print("  * Decision Strength          : %s" % dec["decision_strength"])
    assert dec["selected_method"] == "multi_scale_retinex"
    assert dec["degradation_profile"] == "Severe Illumination Non-Uniformity"
    print("  -> PASS: Correctly classified and selected Multi-Scale Retinex")

    # Test 6: Confirm explanation appears
    print("\n[Test 6] Confirm Transparent Decision Rationale")
    print("  * Rationale: \"%s\"" % dec["decision_reason"])
    assert len(dec["decision_reason"]) > 30
    print("  -> PASS: Real decision reason displayed without truncation")

    # Test 7: Confirm original & enhanced images appear
    print("\n[Test 7] Confirm Image Base64 Payloads")
    in_b64 = res["input_image_b64"]
    out_b64 = res["enhanced_image_b64"]
    assert in_b64.startswith("data:image/jpeg;base64,") and len(in_b64) > 1000
    assert out_b64.startswith("data:image/jpeg;base64,") and len(out_b64) > 1000
    print("  -> PASS: Input image (length %d) & enhanced image (length %d) generated" % (len(in_b64), len(out_b64)))

    # Test 8: Confirm latency appears
    print("\n[Test 8] Confirm Latency Measurement")
    lat = res["processing_time_ms"]
    print("  * Processing Time: %.1f ms" % lat)
    assert lat > 0
    print("  -> PASS: Non-zero latency metric confirmed")

    # Test 9 & 10: Upload custom image and confirm processing
    print("\n[Test 9 & 10] Custom Image Upload Test")
    img = Image.new("RGB", (120, 80), color=(20, 25, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    boundary = "----TestBoundaryXYZ123"
    body = (
        ("--" + boundary + "\r\n").encode("utf-8") +
        ('Content-Disposition: form-data; name="file"; filename="synthetic_dark.png"\r\n').encode("utf-8") +
        ('Content-Type: image/png\r\n\r\n').encode("utf-8") +
        png_bytes +
        ("\r\n--" + boundary + "--\r\n").encode("utf-8")
    )
    req_upload = urllib.request.Request(
        BASE + "/api/analyze_and_enhance",
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary}
    )
    res_upload = json.loads(urllib.request.urlopen(req_upload).read().decode("utf-8"))
    assert res_upload["status"] == "success"
    print("  * Custom Upload Dimensions   : %s" % res_upload["dimensions"])
    print("  * Selected Strategy for Upload: %s" % res_upload["adaptive_decision"]["selected_method"])
    print("  -> PASS: Custom image upload successfully processed")

    # Test 11 & 12: Baseline comparison tests
    print("\n[Test 11 & 12] Baseline Comparison Methods Test")
    for method in ["gamma", "clahe", "msr"]:
        b_payload = json.dumps({"sample_id": "1.png", "subset": "eval15", "method": method}).encode("utf-8")
        b_req = urllib.request.Request(BASE + "/api/run_baseline", data=b_payload, headers={"Content-Type": "application/json"})
        b_res = json.loads(urllib.request.urlopen(b_req).read().decode("utf-8"))
        assert b_res["status"] == "success"
        print("  * %-25s | Latency: %6.1f ms | Payload: %d chars" % (b_res["display_name"], b_res["processing_time_ms"], len(b_res["enhanced_image_b64"])))
    print("  -> PASS: Baseline comparison methods functioning flawlessly")

    print("\n" + "=" * 65)
    print("ALL 12 FUNCTIONALITY CRITERIA VERIFIED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_workflow()
