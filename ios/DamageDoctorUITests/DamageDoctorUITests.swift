import XCTest

/// UI automation for the DamageDoctor iOS app.
/// Pipeline under test: photo input → binary classifier → (if damage) YOLO detector → results UI.
final class DamageDoctorUITests: XCTestCase {

    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments += ["-UITesting"]
        app.launch()
    }

    override func tearDownWithError() throws {
        app = nil
    }

    // MARK: - Clean photo: classifier says no damage, detector UI stays hidden

    func testCleanCarPhotoShowsNoDamageAndSkipsDetector() throws {
        openPhotoPicker()
        selectFixture(named: "clean_car_front")

        let resultBanner = app.staticTexts["classificationResult"]
        XCTAssertTrue(resultBanner.waitForExistence(timeout: 15))
        XCTAssertEqual(resultBanner.label, "No Damage")

        let confidence = app.staticTexts["classificationConfidence"]
        XCTAssertTrue(confidence.exists)
        XCTAssertTrue(confidence.label.contains("%"))

        // Classifier gate: detector panel should not appear for clean images
        XCTAssertFalse(app.otherElements["detectionResults"].exists)
        XCTAssertFalse(app.staticTexts["detectionCount"].exists)
    }

    // MARK: - Damaged photo: classifier passes gate, detector renders labels

    func testDamagedCarPhotoShowsDetections() throws {
        openPhotoPicker()
        selectFixture(named: "cardd_dent_sample")

        let resultBanner = app.staticTexts["classificationResult"]
        XCTAssertTrue(resultBanner.waitForExistence(timeout: 15))
        XCTAssertEqual(resultBanner.label, "Damage Detected")

        let detectionPanel = app.otherElements["detectionResults"]
        XCTAssertTrue(detectionPanel.waitForExistence(timeout: 15))

        let detectionCount = app.staticTexts["detectionCount"]
        XCTAssertTrue(detectionCount.exists)
        XCTAssertNotEqual(detectionCount.label, "0 detections")

        // At least one known damage class from the v7 detector taxonomy
        let expectedLabels = ["Dent", "Scratch", "Crack", "Glass Shatter", "Lamp Broken", "Tire Flat"]
        let foundLabel = expectedLabels.contains { app.staticTexts[$0].exists }
        XCTAssertTrue(foundLabel, "Expected at least one damage type label in the results list")

        // Annotated preview should render bounding boxes over the photo
        let annotatedImage = app.images["annotatedPreview"]
        XCTAssertTrue(annotatedImage.exists)
    }

    // MARK: - Helpers

    private func openPhotoPicker() {
        app.buttons["choosePhotoButton"].tap()
        // Simulator photo library — fixture images seeded in UI test host
        app.buttons["Photo Library"].tap()
    }

    private func selectFixture(named name: String) {
        let photo = app.scrollViews.images[name]
        XCTAssertTrue(photo.waitForExistence(timeout: 5))
        photo.tap()
        app.buttons["Choose"].tap()
    }
}
