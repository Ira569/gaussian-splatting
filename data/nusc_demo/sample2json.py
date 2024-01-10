if __name__ == "__main__":
    import json
    gaussion_camera_json = [{"id": 0, "img_name": "CAM_BACK", "width": 1600, "height": 900,
                            "position": [411.4449780367985, 1181.2631893914647, 0.0],
                            "rotation": [[1,1,1], [2,2,2], [3,3,3]], "fy": 1, "fx": 1},
                            {"id": 1, "img_name": "CAM_BACK", "width": 1600, "height": 900,
                             "position": [411.4449780367985, 1181.2631893914647, 0.0],
                             "rotation": [[1, 1, 1], [2, 2, 2], [3, 3, 3]], "fy": 1, "fx": 1}
                            ]
    json_string = json.dumps(gaussion_camera_json)
    with open("example.json", "w") as json_file:
        json_file.write(json_string)

    print("JSON 文件已生成！")