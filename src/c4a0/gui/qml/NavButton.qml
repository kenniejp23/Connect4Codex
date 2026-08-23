import QtQuick
import QtQuick.Controls

Button {
    id: control
    property bool selected: false
    property string shortLabel: ""
    implicitHeight: 48
    leftPadding: 12
    rightPadding: 12
    hoverEnabled: true

    background: Rectangle {
        radius: 10
        color: control.selected
               ? ApplicationWindow.window.accentMuted
               : (control.hovered ? ApplicationWindow.window.hoverColor : "transparent")
        border.color: control.activeFocus ? ApplicationWindow.window.accentColor : "transparent"
    }

    contentItem: Row {
        spacing: 12
        Rectangle {
            width: 28
            height: 28
            radius: 8
            anchors.verticalCenter: parent.verticalCenter
            color: control.selected ? ApplicationWindow.window.accentColor : ApplicationWindow.window.panelRaisedColor
            Text {
                anchors.centerIn: parent
                text: control.shortLabel
                color: control.selected ? "#111827" : ApplicationWindow.window.mutedTextColor
                font.pixelSize: 12
                font.weight: Font.DemiBold
            }
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: control.text
            color: control.selected ? ApplicationWindow.window.textColor : ApplicationWindow.window.mutedTextColor
            font.pixelSize: 14
            font.weight: control.selected ? Font.DemiBold : Font.Medium
        }
    }
}
