import QtQuick
import QtQuick.Controls

Column {
    property alias title: heading.text
    property alias subtitle: supporting.text
    spacing: 4

    Text {
        id: heading
        width: parent.width
        color: ApplicationWindow.window.textColor
        font.pixelSize: 21
        font.weight: Font.DemiBold
    }
    Text {
        id: supporting
        width: parent.width
        color: ApplicationWindow.window.mutedTextColor
        font.pixelSize: 13
        wrapMode: Text.WordWrap
    }
}
